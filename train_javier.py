#!/usr/bin/env python3

import os
import torch
import gc
import shutil
import time
from copy import deepcopy

from DataLoader.cohorts import all_cohorts

from Helpers.args import get_args_training
from Helpers.utils import seed_everything, logger_setup
from Helpers.data_utils import create_cv_splits

from Trainers.unet_trainer3d import Trainer3D
from Trainers.unet_trainer2d import Trainer2D


if __name__ == '__main__':

    config = get_args_training()
    if config.network != "unet_tcm_trans": seed_everything(config.seed)

    if config.views == "3D":
        logger = logger_setup(folder=os.path.join(config.log_folder, "3D", config.label_method)) 
    elif config.views == "2D":
        logger = logger_setup(folder=os.path.join(config.log_folder, "2D", config.label_method)) 

    if config.n_cross_val == None:
        
        fixed_sets = config.fixed_sets

        if config.views == "3D":
            trainer = Trainer3D(fixed_sets, all_cohorts, config, logger)
        elif config.views == "2D":
            trainer = Trainer2D(fixed_sets, all_cohorts, config, logger)

        logger.info(f"Arguments:\n {trainer.config}")
        logger.info(f"Results will be stored in directory: {trainer.output_folder}")
        
        logger.info("TRAINING STARTING...")
        trained_net = trainer.train()
        
        
        logger.info("TEST STARTING...")
        trainer.test(mode="test")
        logger.info("TEST FINISHED!!")
    
    else:
        config, tmp_folder = create_cv_splits(config, logger)
        logger.info(f"Starting Cross Validation. Number of folds: {config.n_cross_val}...")
        start_time = time.time()
        val_loss = []
        for i in range(config.n_cross_val):
            config_cv = deepcopy(config)
            fixed_sets = os.path.join(config_cv.fixed_sets, f"data_cv_{i}.json")
            config_cv.fixed_sets = fixed_sets
            config_cv.use_predifined_sets = True
            if config_cv.views == "3D":
                logger.info("####################")
                logger.info(f"#    CV SPLIT {i}    #")
                logger.info("####################")
                trainer = Trainer3D(fixed_sets, all_cohorts, config_cv, logger)
            elif config_cv.views == "2D":
                logger.info("####################")
                logger.info(f"#    CV SPLIT {i}    #")
                logger.info("####################")
                trainer = Trainer2D(fixed_sets, all_cohorts, config_cv, logger)

            logger.info(f"Arguments:\n {trainer.config}")
            logger.info(f"Results will be stored in directory: {trainer.output_folder}")
            
            logger.info("TRAINING STARTING...")
            trained_net = trainer.train()
            output_folder = trainer.output_folder
            val_loss.append(min(trainer.val_loss))

            logger.info("TEST STARTING...")
            trainer.test(mode="test")
            logger.info("TEST FINISHED!!")

            # Perform test on best model after the last CV training
            if i == config.n_cross_val -1:
                best_cv = val_loss.index(min(val_loss))
                logger.info(f"Best CV split: {best_cv}")
                # output_folder = output_folder[:-1] + str(best_cv)
                # trainer.net.load_state_dict(torch.load(os.path.join(output_folder, "network")))

                # logger.info("TEST STARTING...")
                # trainer.test(mode="test")
                # logger.info("TEST FINISHED!!")

            del trainer, config_cv
            gc.collect()
            torch.cuda.empty_cache()
        cv_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        logger.info(f"Cross Validation Finished! It took {cv_time}")
        shutil.rmtree(tmp_folder)
            