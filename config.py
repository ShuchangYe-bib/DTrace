from packages import ArgParser

args = ArgParser()

########################
# environment settings #
########################
args.seed = 0
args.device = "cuda"
args.n_gpu = 1

####################
# dataset settings #
####################
args.num_images = 1 # the number of images per patient
args.dataset_name = "mimic_cxr" # the dataset to be used.
args.image_shape = (3, 224, 224) # the shape of the images <c, w, h>
# inference
if args.dataset_name == "iu_xray":
	args.image_dir = "data/iu_xray/images/"
	args.ann_path = "data/iu_xray/annotation.json"
elif args.dataset_name == "mimic_cxr":
	args.image_dir = "data/mimic_cxr/images/"
	args.ann_path = "data/mimic_cxr/annotation.json"

#####################
# training settings #
#####################
args.batch_size = 16 # the number of samples for a batch
args.epochs = 3 # the number of training epochs.
args.save_dir = f"results/{args.dataset_name}" # the path to save the models
args.record_dir = "records/" # the path to save the results of experiment
args.save_period = 1 # the saving period
args.model_path = f"results/{args.dataset_name}/model_best.pth" # path to the loaded pre-trained model
args.load_path = f"results/{args.dataset_name}/model_best.pth" # path to the loaded pre-trained model
args.monitor_mode = "max" # whether to max or min the metric (choices=['min', 'max'])
args.monitor_metric = "BLEU_4" # the metric to be monitored e.g. BLEU_4, score
args.early_stop = 50 # the patience of training
args.resume = None


#######################
# dataloader settings #
#######################
args.max_seq_length = 256 # the maximum sequence length of the reports.
args.threshold = 3 # the cut off frequency for the words.
args.num_workers = 16 # the number of workers for dataloader.


##################
# model settings #
##################
# transformer
args.visual_extractor = 'patchify' # the visual extractor to be used
args.mae_checkpoint = "weights/mae_visualize_vit_base.pth" # whether to load the pretrained mae. choose from ["mae_visualize_vit_base.pth", "weights/mae_pretrain_vit_base.pth", None]           
args.encoder = "vit" # the encoder to be used.
args.d_model = 768 # the dimension of Transformer
args.d_ff = 3072 # the dimension of FFN
args.patch_size = 16 # the size of each patch
args.num_patches = 14*14
args.d_vf = 768 # the dimension of the patch features
args.num_heads = 8 # the number of heads in Transformer
args.num_layers = 6 # the number of layers of Transformer
args.dropout = 0.1 # the dropout rate of Transformer.
args.logit_layers = 1 # the number of the logit layer
args.bos_idx = 0 # the index of <bos>
args.eos_idx = 0 # the index of <eos>
args.pad_idx = 0 # the index of <pad>
args.use_bn = 0 # whether to use batch normalization
args.drop_prob_lm = 0.5 # the dropout rate of the output layer.')
# memory
args.rm_num_slots = 3 # the number of memory slots
args.rm_num_heads = 8 # the numebr of heads in rm
args.rm_d_model = args.d_model # the dimension of rm
# head
args.num_classes = 14 # the number of labels of radiology diagnosis

###################
# sample settings #
###################
args.mask_ratio = "dynamic" # fixed or dynamic
args.img_mask_rate = 0.75
args.txt_mask_rate = 0.75
args.sample_method = 'beam_search' # the sample methods to sample a report
args.beam_size = 3 # the beam size when beam searching
args.temperature = 1.0 # the temperature when sampling
args.sample_n = 1 # the sample number per image
args.group_size = 1 # the group size
args.output_logsoftmax = 1 # whether to output the probabilities
args.decoding_constraint = 0 # whether decoding constraint
args.block_trigrams = 1 # whether to use block trigrams

#########################
# optimization settings #
#########################
# optimizer
args.loss_fn = 'Hybrid' # the loss function for gradient descent
args.optim = 'AdamW' # the type of the optimizer.
args.lr_ve = 5e-4 # the learning rate for the visual extractor
args.lr_ed = 1e-3 # the learning rate for the remaining parameters
args.weight_decay = 5e-5 # the weight decay
args.amsgrad = True
# learning rate
args.lr_scheduler = 'StepLR' # the type of the learning rate scheduler
args.step_size = 100 # the step size of the learning rate scheduler
args.gamma = 0.1 # the gamma of the learning rate scheduler

##########################
# visualization settings #
##########################
args.image_path = "data/iu_xray/images/CXR1_1_IM-0001/1.png" # path to the x-ray image
args.n_sample = 1 # the number of attention map to plot, -1 for plot all


