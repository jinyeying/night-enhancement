from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter

##################################################################################################
##################################################################################################
"""
Do you experiment setup here!
"""
# Dataset directory
data_name   = 'ExDark'
data_dir    = '/home/Datasets/ExDark_Dataset/processed/'
# Model name
model_name  = 'vgg16'
# Number of classes in the dataset
num_classes = 12
# Batch size for training (default = 8)
batch_size  = 16
# Number of epochs to train for
num_epochs  = 60
# Flag for feature extracting. 
# When False, we finetune the whole model,
# When True, we only update the reshaped layer params
feature_extract = False


# Checkpoint directory
ftrextsuffix= 'True' if feature_extract else 'False'
savename    = model_name + '_' + 'featureextract' + ftrextsuffix + '_' + data_name
save_dir    = './ckpts/' + savename + '/'
##################################################################################################
##################################################################################################
"""
Function to create the model
"""
def initialize_model(model_name, 
					 num_classes, 
					 feature_extract, 
					 use_pretrained=True):
	# Initialize these variables which will be set in this if statement. Each of these
	# variables is model specific.
	model      = None
	input_size = 0
	# Set which params to train
	def set_parameter_requires_grad(model, 
									feature_extracting):
		if feature_extracting:
			for param in model.features.parameters():
				param.requires_grad = False
	# Create the model here
	if model_name == "vgg16":
		""" VGG16
		"""
		model = models.vgg16(pretrained=use_pretrained)
		set_parameter_requires_grad(model, feature_extract)
		num_ftrs = model.classifier[6].in_features
		model.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224
	else:
		print("Invalid model name, exiting...")
		exit()
	return model, input_size


##################################################################################################
##################################################################################################
"""
Function to train and validate the model
"""
def train_model(model, 
				dataloaders, 
				criterion, 
				optimizer, 
				logger,
				num_epochs=25, 
				is_inception=False):
	# Initialize some variables
	since           = time.time()
	val_acc_history = []
	best_model_wts  = copy.deepcopy(model.state_dict())
	best_acc        = 0.0
	# Train and validate for the given number of epochs
	glb_iter = 0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss     = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.cuda()
				labels = labels.cuda()

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# Get model outputs and calculate loss
					# Special case for inception because in training it has an auxiliary output. In train
					#   mode we calculate the loss by summing the final output and the auxiliary output
					#   but in testing we only consider the final output.
					if is_inception and phase == 'train':
						# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
						outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
						loss2 = criterion(aux_outputs, labels)
						loss  = loss1 + 0.4*loss2
					else:
						outputs = model(inputs)
						loss    = criterion(outputs, labels)

					_, preds = torch.max(outputs, 1)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				glb_iter         += 1
				running_loss     += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
				if phase == 'train':
					logger.add_scalar('Train/iter_loss', loss.item(), glb_iter)
				elif phase == 'val':
					logger.add_scalar('Val/iter_loss', loss.item(), glb_iter)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc  = running_corrects.double() / len(dataloaders[phase].dataset)
			if phase == 'train':
				logger.add_scalar('Train/epoch_loss', epoch_loss, epoch)
				logger.add_scalar('Train/epoch_acc', epoch_acc, epoch)
			elif phase == 'val':
				logger.add_scalar('Val/epoch_loss', epoch_loss, epoch)
				logger.add_scalar('Val/epoch_acc', epoch_acc, epoch)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			# Deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
			if phase == 'val':
				val_acc_history.append(epoch_acc)

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, val_acc_history

##################################################################################################
##################################################################################################
if __name__ == '__main__':
	
	# Directories
	logs_dir    = save_dir + '/logs/'
	nets_dir    = save_dir + '/nets/'
	os.makedirs(logs_dir, exist_ok=True)
	os.makedirs(nets_dir, exist_ok=True)
	
	# Initialize log writer
	logger= SummaryWriter(logs_dir)
	
	# Create model
	model, input_size = initialize_model(model_name, 
										 num_classes, 
										 feature_extract, 
										 use_pretrained=True)
	# Use all the 4 GPUs by default
	model = nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()

	# Create optimizer
	params_to_update = model.parameters()
	print("Params to learn:")
	if feature_extract:
		params_to_update = []
		for name,param in model.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
				print("\t",name)
	else:
		for name,param in model.named_parameters():
			if param.requires_grad == True:
				print("\t",name)
	optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

	# Create train and val loaders
	# Data augmentation and normalization for training
	# Just normalization for validation
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(input_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}


	# Create training and validation datasets
	image_datasets   = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
						data_transforms[x]) for x in ['train', 'val']}
	# Create training and validation dataloaders
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
						batch_size=batch_size, 
						shuffle=True and (x == 'train'), 
						num_workers=16) for x in ['train', 'val']}
	

	# Setup the loss fxn
	criterion = nn.CrossEntropyLoss()

	# Train and evaluate
	model_bst, val_hist = train_model(model, 
									  dataloaders_dict, 
									  criterion, 
									  optimizer, 
									  logger,
									  num_epochs=num_epochs, 
									  is_inception=(model_name=="inception"))
	# Save the best model
	savefilename = nets_dir+'/model_best.tar'
	torch.save(model.state_dict(), savefilename)
