from .gcs_client import GCSclient
from .opt_callbacks import optimizer_h5,NBatchProgBarLogger,NEpochModelCheckpoint_wOptimizer
from .layers_loss_metrics import GeMPoolingLayer,ArcFace,ArcFaceLoss_Metrics
from .dataset_parser import Lookup,Preprocess
from .models import Efn_Gem_Arc_builder,Branches_builder,Transfer_builder

import tensorflow as tf
if tf.__version__>="2.2.0":
    from .models import Model_w_self_backpropagated_branches