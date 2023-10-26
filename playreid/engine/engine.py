# encoding: utf-8
"""
@author:  lorenzo
@contact: baiyingpoi123@gmail.com
"""
import os
import logging
import torch
from collections import OrderedDict

from playreid.data import build_reid_test_loader
from playreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from playreid.evaluation.testing import flatten_results_dict
from playreid.solver import build_lr_scheduler, build_optimizer
from playreid.utils import comm
from playreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from playreid.utils.collect_env import collect_env_info
from playreid.utils.env import seed_all_rng
from playreid.utils.file_io import PathManager
from playreid.utils.logger import setup_logger
from playreid.utils.params import ContiguousParams
from playreid.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)

__all__ = ["default_setup", "do_train", "do_test", "auto_scale_hyperparams"]


def default_setup(cfg, args, log_name=None):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
        log_name (str): the log file name, it will be 'log.txt' by default
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    # setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    if log_name is None:
        logger = setup_logger(output_dir, distributed_rank=rank)
    else:
        logger = setup_logger(os.path.join(output_dir, log_name), distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng()

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def get_evaluator(cfg, dataset_name, output_dir=None):
    data_loader, num_query = build_reid_test_loader(cfg, dataset_name=dataset_name)
    return data_loader, ReidEvaluator(cfg, num_query, output_dir)


def do_train(cfg, model, data_loader, resume=False, qat=False):
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    data_loader_iter = iter(data_loader)
    model.train()
    optimizer, param_wrapper = build_optimizer(cfg, model, contiguous=False if qat == True else True)
    iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)
    checkpointer = Checkpointer(
        model,
        cfg.OUTPUT_DIR,
        save_to_disk=comm.is_main_process(),
        optimizer=optimizer,
        **scheduler
    )

    start_epoch = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("epoch", -1) + 1)
    iteration = start_iter = start_epoch * iters_per_epoch
    
    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    delay_epochs = cfg.SOLVER.DELAY_EPOCHS

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_epoch)
    if len(cfg.DATASETS.TESTS) == 1:
        metric_name = "metric"
    else:
        metric_name = cfg.DATASETS.TESTS[0] + "/metric"
    
    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR)
        ]
        if comm.is_main_process()
        else []
    )

    logger.info("Start training from epoch {}".format(start_epoch))
    with EventStorage(start_iter) as storage:
        for epoch in range(start_epoch, max_epoch):
            storage.epoch = epoch
            # print(f"{epoch}, {optimizer.param_groups[0]['lr']}")
            for _ in range(iters_per_epoch):
                data = next(data_loader_iter)
                storage.iter = iteration

                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                
                if isinstance(param_wrapper, ContiguousParams):
                    param_wrapper.assert_buffer_is_valid()
                if iteration - start_iter > 5 and \
                        ((iteration + 1) % 200 == 0 or iteration == max_iter - 1) and \
                        ((iteration + 1) % iters_per_epoch != 0):
                    for writer in writers:
                        writer.write()
                
                iteration += 1

                if iteration <= warmup_iters:
                        scheduler["warmup_sched"].step()
            
            # Write metrics after each epoch
            for writer in writers:
                writer.write()

            if iteration > warmup_iters and (epoch + 1) > delay_epochs:
                    scheduler["lr_sched"].step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                results = do_test(cfg, model)

                # add validation metrics at each epoch, the test results are not dumped to EventStorage
                # if comm.is_main_process():
                #     for k in results:
                #         writers[-1]._writer.add_scalar(k, results[k], epoch)
            else:
                results = {}
            flatten_results = flatten_results_dict(results)

            metric_dict = dict(metric=flatten_results[metric_name] if metric_name in flatten_results else -1)
            periodic_checkpointer.step(epoch, **metric_dict)


def do_test(cfg, model):
    logger = logging.getLogger(__name__)
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        logger.info("Prepare testing set")
        try:
            data_loader, evaluator = get_evaluator(cfg, dataset_name)
        except NotImplementedError:
            logger.warn("No evaluator found. implement its `build_evaluator` method.")
            results[dataset_name] = {}
            continue
        results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
        results[dataset_name] = results_i

        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(results)
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            results_i['dataset'] = dataset_name
            print_csv_format(results_i)
    
    if len(results) == 1:
        results = list(results.values())[0]

    return results


def auto_scale_hyperparams(cfg, num_classes):
    """
    This is used for auto-computation actual training iterations,
    because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
    so we need to convert specific hyper-param to training iterations.
    """
    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()

    # If you don't hard-code the number of classes, it will compute the number automatically
    if cfg.MODEL.HEADS.NUM_CLASSES == 0:
        output_dir = cfg.OUTPUT_DIR
        cfg.MODEL.HEADS.NUM_CLASSES = num_classes
        logger = logging.getLogger(__name__)
        logger.info(f"Auto-scaling the num_classes={cfg.MODEL.HEADS.NUM_CLASSES}")

        # Update the saved config file to make the number of classes valid
        if comm.is_main_process() and output_dir:
            # Note: some of our scripts may expect the existence of
            # config.yaml in output directory
            path = os.path.join(output_dir, "config.yaml")
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())

    if frozen: cfg.freeze()

    return cfg