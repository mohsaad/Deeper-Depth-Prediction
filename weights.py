# Mohammad Saad
# 2/24/2018
# weights.py
# Loads weights into the model
# Pretty much the same as @iapatil's version

import numpy as np
import torch

def load_weights(model, filename, dtype):

	model_params = model.state_dict()
	data_dict = np.load(weights_file, encoding='latin1').item()

	# initial layer
	model_params["conv1.weight"] = torch.from_numpy(data_dict["conv1"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["conv1.bias"] = torch.from_numpy(data_dict["conv1"]["weights"]).type(dtype)
	model_params["bn1.weight"] = torch.from_numpy(data_dict["bn_conv1"]["scale"]).type(dtype)
	model_params["bn1.bias"] = torch.from_numpy(data_dict["bn_conv1"]["offset"]).type(dtype)


	## first projection layer
	model_params["proj1.conv1.weight"] = torch.from_numpy(data_dict["res2a_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj1.bn1.weight"] = torch.from_numpy(data_dict["bn2a_branch2a"]["scale"]).type(dtype)
	model_params["proj1.bn1.bias"] = torch.from_numpy(data_dict["bn2a_branch2a"]["offset"]).type(dtype)

	model_params["proj1.conv2.weight"] = torch.from_numpy(data_dict["res2a_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj1.bn2.weight"] = torch.from_numpy(data_dict["bn2a_branch2b"]["scale"]).type(dtype)
	model_params["proj1.bn2.bias"] = torch.from_numpy(data_dict["bn2a_branch2b"]["offset"]).type(dtype)

	model_params["proj1.conv3.weight"] = torch.from_numpy(data_dict["res2a_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj1.bn3.weight"] = torch.from_numpy(data_dict["bn2a_branch2c"]["scale"]).type(dtype)
	model_params["proj1.bn3.bias"] = torch.from_numpy(data_dict["bn2a_branch2c"]["offset"]).type(dtype)

	model_params["proj1.conv4.weight"] = torch.from_numpy(data_dict["res2a_branch1"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj1.bn4.weight"] = torch.from_numpy(data_dict["bn2a_branch1"]["scale"]).type(dtype)
	model_params["proj1.bn4.bias"] = torch.from_numpy(data_dict["bn2a_branch1"]["offset"]).type(dtype)

	#
	model_params["res1_1.conv1.weight"] = torch.from_numpy(data_dict["res2b_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res1_1.bn1.weight"] = torch.from_numpy(data_dict["bn2b_branch2a"]["scale"]).type(dtype)
	model_params["res1_1.bn1.bias"] = torch.from_numpy(data_dict["bn2b_branch2a"]["offset"]).type(dtype)

	model_params["res1_1.conv2.weight"] = torch.from_numpy(data_dict["res2b_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res1_1.bn2.weight"] = torch.from_numpy(data_dict["bn2b_branch2b"]["scale"]).type(dtype)
	model_params["res1_1.bn2.bias"] = torch.from_numpy(data_dict["bn2b_branch2b"]["offset"]).type(dtype)

	model_params["res1_1.conv3.weight"] = torch.from_numpy(data_dict["res2b_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res1_1.bn3.weight"] = torch.from_numpy(data_dict["bn2b_branch2c"]["scale"]).type(dtype)
	model_params["res1_1.bn3.bias"] = torch.from_numpy(data_dict["bn2b_branch2c"]["offset"]).type(dtype)

	#
	model_params["res1_2.conv1.weight"] = torch.from_numpy(data_dict["res2c_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res1_2.bn1.weight"] = torch.from_numpy(data_dict["bn2c_branch2a"]["scale"]).type(dtype)
	model_params["res1_2.bn1.bias"] = torch.from_numpy(data_dict["bn2c_branch2a"]["offset"]).type(dtype)

	model_params["res1_2.conv2.weight"] = torch.from_numpy(data_dict["res2c_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res1_2.bn2.weight"] = torch.from_numpy(data_dict["bn2c_branch2b"]["scale"]).type(dtype)
	model_params["res1_2.bn2.bias"] = torch.from_numpy(data_dict["bn2c_branch2b"]["offset"]).type(dtype)

	model_params["res1_2.conv3.weight"] = torch.from_numpy(data_dict["res2c_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res1_2.bn3.weight"] = torch.from_numpy(data_dict["bn2c_branch2c"]["scale"]).type(dtype)
	model_params["res1_2.bn3.bias"] = torch.from_numpy(data_dict["bn2c_branch2c"]["offset"]).type(dtype)

	## second projection layer
	model_params["proj2.conv1.weight"] = torch.from_numpy(data_dict["res3a_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj2.bn1.weight"] = torch.from_numpy(data_dict["bn3a_branch2a"]["scale"]).type(dtype)
	model_params["proj2.bn1.bias"] = torch.from_numpy(data_dict["bn3a_branch2a"]["offset"]).type(dtype)

	model_params["proj2.conv2.weight"] = torch.from_numpy(data_dict["res3a_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj2.bn2.weight"] = torch.from_numpy(data_dict["bn3a_branch2b"]["scale"]).type(dtype)
	model_params["proj2.bn2.bias"] = torch.from_numpy(data_dict["bn3a_branch2b"]["offset"]).type(dtype)

	model_params["proj2.conv3.weight"] = torch.from_numpy(data_dict["res3a_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj2.bn3.weight"] = torch.from_numpy(data_dict["bn3a_branch2c"]["scale"]).type(dtype)
	model_params["proj2.bn3.bias"] = torch.from_numpy(data_dict["bn3a_branch2c"]["offset"]).type(dtype)

	model_params["proj2.conv4.weight"] = torch.from_numpy(data_dict["res3a_branch1"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj2.bn4.weight"] = torch.from_numpy(data_dict["bn3a_branch1"]["scale"]).type(dtype)
	model_params["proj2.bn4.bias"] = torch.from_numpy(data_dict["bn3a_branch1"]["offset"]).type(dtype)

	#
	model_params["res2_1.conv1.weight"] = torch.from_numpy(data_dict["res3b_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_1.bn1.weight"] = torch.from_numpy(data_dict["bn3b_branch2a"]["scale"]).type(dtype)
	model_params["res2_1.bn1.bias"] = torch.from_numpy(data_dict["bn3b_branch2a"]["offset"]).type(dtype)

	model_params["res2_1.conv2.weight"] = torch.from_numpy(data_dict["res3b_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_1.bn2.weight"] = torch.from_numpy(data_dict["bn3b_branch2b"]["scale"]).type(dtype)
	model_params["res2_1.bn2.bias"] = torch.from_numpy(data_dict["bn3b_branch2b"]["offset"]).type(dtype)

	model_params["res2_1.conv3.weight"] = torch.from_numpy(data_dict["res3b_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_1.bn3.weight"] = torch.from_numpy(data_dict["bn3b_branch2c"]["scale"]).type(dtype)
	model_params["res2_1.bn3.bias"] = torch.from_numpy(data_dict["bn3b_branch2c"]["offset"]).type(dtype)

	#
	model_params["res2_2.conv1.weight"] = torch.from_numpy(data_dict["res3c_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_2.bn1.weight"] = torch.from_numpy(data_dict["bn3c_branch2a"]["scale"]).type(dtype)
	model_params["res2_2.bn1.bias"] = torch.from_numpy(data_dict["bn3c_branch2a"]["offset"]).type(dtype)

	model_params["res2_2.conv2.weight"] = torch.from_numpy(data_dict["res3c_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_2.bn2.weight"] = torch.from_numpy(data_dict["bn3c_branch2b"]["scale"]).type(dtype)
	model_params["res2_2.bn2.bias"] = torch.from_numpy(data_dict["bn3c_branch2b"]["offset"]).type(dtype)

	model_params["res2_2.conv3.weight"] = torch.from_numpy(data_dict["res3c_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_2.bn3.weight"] = torch.from_numpy(data_dict["bn3c_branch2c"]["scale"]).type(dtype)
	model_params["res2_2.bn3.bias"] = torch.from_numpy(data_dict["bn3c_branch2c"]["offset"]).type(dtype)

	#
	model_params["res2_3.conv1.weight"] = torch.from_numpy(data_dict["res3d_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_3.bn1.weight"] = torch.from_numpy(data_dict["bn3d_branch2a"]["scale"]).type(dtype)
	model_params["res2_3.bn1.bias"] = torch.from_numpy(data_dict["bn3d_branch2a"]["offset"]).type(dtype)

	model_params["res2_3.conv2.weight"] = torch.from_numpy(data_dict["res3d_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_3.bn2.weight"] = torch.from_numpy(data_dict["bn3d_branch2b"]["scale"]).type(dtype)
	model_params["res2_3.bn2.bias"] = torch.from_numpy(data_dict["bn3d_branch2b"]["offset"]).type(dtype)

	model_params["res2_3.conv3.weight"] = torch.from_numpy(data_dict["res3d_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res2_3.bn3.weight"] = torch.from_numpy(data_dict["bn3d_branch2c"]["scale"]).type(dtype)
	model_params["res2_3.bn3.bias"] = torch.from_numpy(data_dict["bn3d_branch2c"]["offset"]).type(dtype)

	## third projection layer
	model_params["proj3.conv1.weight"] = torch.from_numpy(data_dict["res4a_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj3.bn1.weight"] = torch.from_numpy(data_dict["bn4a_branch2a"]["scale"]).type(dtype)
	model_params["proj3.bn1.bias"] = torch.from_numpy(data_dict["bn4a_branch2a"]["offset"]).type(dtype)

	model_params["proj3.conv2.weight"] = torch.from_numpy(data_dict["res4a_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj3.bn2.weight"] = torch.from_numpy(data_dict["bn4a_branch2b"]["scale"]).type(dtype)
	model_params["proj3.bn2.bias"] = torch.from_numpy(data_dict["bn4a_branch2b"]["offset"]).type(dtype)

	model_params["proj3.conv3.weight"] = torch.from_numpy(data_dict["res4a_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj3.bn3.weight"] = torch.from_numpy(data_dict["bn4a_branch2c"]["scale"]).type(dtype)
	model_params["proj3.bn3.bias"] = torch.from_numpy(data_dict["bn4a_branch2c"]["offset"]).type(dtype)

	model_params["proj3.conv4.weight"] = torch.from_numpy(data_dict["res4a_branch1"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj3.bn4.weight"] = torch.from_numpy(data_dict["bn4a_branch1"]["scale"]).type(dtype)
	model_params["proj3.bn4.bias"] = torch.from_numpy(data_dict["bn4a_branch1"]["offset"]).type(dtype)

	#
	model_params["res3_1.conv1.weight"] = torch.from_numpy(data_dict["res4b_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_1.bn1.weight"] = torch.from_numpy(data_dict["bn4b_branch2a"]["scale"]).type(dtype)
	model_params["res3_1.bn1.bias"] = torch.from_numpy(data_dict["bn4b_branch2a"]["offset"]).type(dtype)

	model_params["res3_1.conv2.weight"] = torch.from_numpy(data_dict["res4b_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_1.bn2.weight"] = torch.from_numpy(data_dict["bn4b_branch2b"]["scale"]).type(dtype)
	model_params["res3_1.bn2.bias"] = torch.from_numpy(data_dict["bn4b_branch2b"]["offset"]).type(dtype)

	model_params["res3_1.conv3.weight"] = torch.from_numpy(data_dict["res4b_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_1.bn3.weight"] = torch.from_numpy(data_dict["bn4b_branch2c"]["scale"]).type(dtype)
	model_params["res3_1.bn3.bias"] = torch.from_numpy(data_dict["bn4b_branch2c"]["offset"]).type(dtype)

	#
	model_params["res3_2.conv1.weight"] = torch.from_numpy(data_dict["res4c_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_2.bn1.weight"] = torch.from_numpy(data_dict["bn4c_branch2a"]["scale"]).type(dtype)
	model_params["res3_2.bn1.bias"] = torch.from_numpy(data_dict["bn4c_branch2a"]["offset"]).type(dtype)

	model_params["res3_2.conv2.weight"] = torch.from_numpy(data_dict["res4c_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_2.bn2.weight"] = torch.from_numpy(data_dict["bn4c_branch2b"]["scale"]).type(dtype)
	model_params["res3_2.bn2.bias"] = torch.from_numpy(data_dict["bn4c_branch2b"]["offset"]).type(dtype)

	model_params["res3_2.conv3.weight"] = torch.from_numpy(data_dict["res4c_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_2.bn3.weight"] = torch.from_numpy(data_dict["bn4c_branch2c"]["scale"]).type(dtype)
	model_params["res3_2.bn3.bias"] = torch.from_numpy(data_dict["bn4c_branch2c"]["offset"]).type(dtype)

	#
	model_params["res3_3.conv1.weight"] = torch.from_numpy(data_dict["res4d_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_3.bn1.weight"] = torch.from_numpy(data_dict["bn4d_branch2a"]["scale"]).type(dtype)
	model_params["res3_3.bn1.bias"] = torch.from_numpy(data_dict["bn4d_branch2a"]["offset"]).type(dtype)

	model_params["res3_3.conv2.weight"] = torch.from_numpy(data_dict["res4d_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_3.bn2.weight"] = torch.from_numpy(data_dict["bn4d_branch2b"]["scale"]).type(dtype)
	model_params["res3_3.bn2.bias"] = torch.from_numpy(data_dict["bn4d_branch2b"]["offset"]).type(dtype)

	model_params["res3_3.conv3.weight"] = torch.from_numpy(data_dict["res4d_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_3.bn3.weight"] = torch.from_numpy(data_dict["bn4d_branch2c"]["scale"]).type(dtype)
	model_params["res3_3.bn3.bias"] = torch.from_numpy(data_dict["bn4d_branch2c"]["offset"]).type(dtype)

	#
	model_params["res3_4.conv1.weight"] = torch.from_numpy(data_dict["res4e_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_4.bn1.weight"] = torch.from_numpy(data_dict["bn4e_branch2a"]["scale"]).type(dtype)
	model_params["res3_4.bn1.bias"] = torch.from_numpy(data_dict["bn4e_branch2a"]["offset"]).type(dtype)

	model_params["res3_4.conv2.weight"] = torch.from_numpy(data_dict["res4e_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_4.bn2.weight"] = torch.from_numpy(data_dict["bn4e_branch2b"]["scale"]).type(dtype)
	model_params["res3_4.bn2.bias"] = torch.from_numpy(data_dict["bn4e_branch2b"]["offset"]).type(dtype)

	model_params["res3_4.conv3.weight"] = torch.from_numpy(data_dict["res4e_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_4.bn3.weight"] = torch.from_numpy(data_dict["bn4e_branch2c"]["scale"]).type(dtype)
	model_params["res3_4.bn3.bias"] = torch.from_numpy(data_dict["bn4e_branch2c"]["offset"]).type(dtype)

	#
	model_params["res3_5.conv1.weight"] = torch.from_numpy(data_dict["res4f_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_5.bn1.weight"] = torch.from_numpy(data_dict["bn4f_branch2a"]["scale"]).type(dtype)
	model_params["res3_5.bn1.bias"] = torch.from_numpy(data_dict["bn4f_branch2a"]["offset"]).type(dtype)

	model_params["res3_5.conv2.weight"] = torch.from_numpy(data_dict["res4f_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_5.bn2.weight"] = torch.from_numpy(data_dict["bn4f_branch2b"]["scale"]).type(dtype)
	model_params["res3_5.bn2.bias"] = torch.from_numpy(data_dict["bn4f_branch2b"]["offset"]).type(dtype)

	model_params["res3_5.conv3.weight"] = torch.from_numpy(data_dict["res4f_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res3_5.bn3.weight"] = torch.from_numpy(data_dict["bn4f_branch2c"]["scale"]).type(dtype)
	model_params["res3_5.bn3.bias"] = torch.from_numpy(data_dict["bn4f_branch2c"]["offset"]).type(dtype)

	## fourth projection layer
	model_params["proj4.conv1.weight"] = torch.from_numpy(data_dict["res5a_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj4.bn1.weight"] = torch.from_numpy(data_dict["bn5a_branch2a"]["scale"]).type(dtype)
	model_params["proj4.bn1.bias"] = torch.from_numpy(data_dict["bn5a_branch2a"]["offset"]).type(dtype)

	model_params["proj4.conv2.weight"] = torch.from_numpy(data_dict["res5a_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj4.bn2.weight"] = torch.from_numpy(data_dict["bn5a_branch2b"]["scale"]).type(dtype)
	model_params["proj4.bn2.bias"] = torch.from_numpy(data_dict["bn5a_branch2b"]["offset"]).type(dtype)

	model_params["proj4.conv3.weight"] = torch.from_numpy(data_dict["res5a_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj4.bn3.weight"] = torch.from_numpy(data_dict["bn5a_branch2c"]["scale"]).type(dtype)
	model_params["proj4.bn3.bias"] = torch.from_numpy(data_dict["bn5a_branch2c"]["offset"]).type(dtype)

	model_params["proj4.conv4.weight"] = torch.from_numpy(data_dict["res5a_branch1"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["proj4.bn4.weight"] = torch.from_numpy(data_dict["bn5a_branch1"]["scale"]).type(dtype)
	model_params["proj4.bn4.bias"] = torch.from_numpy(data_dict["bn5a_branch1"]["offset"]).type(dtype)

	#
	model_params["res4_1.conv1.weight"] = torch.from_numpy(data_dict["res5b_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res4_1.bn1.weight"] = torch.from_numpy(data_dict["bn5b_branch2a"]["scale"]).type(dtype)
	model_params["res4_1.bn1.bias"] = torch.from_numpy(data_dict["bn5b_branch2a"]["offset"]).type(dtype)

	model_params["res4_1.conv2.weight"] = torch.from_numpy(data_dict["res5b_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res4_1.bn2.weight"] = torch.from_numpy(data_dict["bn5b_branch2b"]["scale"]).type(dtype)
	model_params["res4_1.bn2.bias"] = torch.from_numpy(data_dict["bn5b_branch2b"]["offset"]).type(dtype)

	model_params["res4_1.conv3.weight"] = torch.from_numpy(data_dict["res5b_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res4_1.bn3.weight"] = torch.from_numpy(data_dict["bn5b_branch2c"]["scale"]).type(dtype)
	model_params["res4_1.bn3.bias"] = torch.from_numpy(data_dict["bn5b_branch2c"]["offset"]).type(dtype)

	#
	model_params["res4_2.conv1.weight"] = torch.from_numpy(data_dict["res5c_branch2a"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res4_2.bn1.weight"] = torch.from_numpy(data_dict["bn5c_branch2a"]["scale"]).type(dtype)
	model_params["res4_2.bn1.bias"] = torch.from_numpy(data_dict["bn5c_branch2a"]["offset"]).type(dtype)

	model_params["res4_2.conv2.weight"] = torch.from_numpy(data_dict["res5c_branch2b"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res4_2.bn2.weight"] = torch.from_numpy(data_dict["bn5c_branch2b"]["scale"]).type(dtype)
	model_params["res4_2.bn2.bias"] = torch.from_numpy(data_dict["bn5c_branch2b"]["offset"]).type(dtype)

	model_params["res4_2.conv3.weight"] = torch.from_numpy(data_dict["res5c_branch2c"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["res4_2.bn3.weight"] = torch.from_numpy(data_dict["bn5c_branch2c"]["scale"]).type(dtype)
	model_params["res4_2.bn3.bias"] = torch.from_numpy(data_dict["bn5c_branch2c"]["offset"]).type(dtype)

	## Middle convolution layer
	model_params["conv2.weight"] = torch.from_numpy(data_dict["conv2"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["conv2.bias"] = torch.from_numpy(data_dict["conv2"]["weights"]).type(dtype)
	model_params["bn2.weight"] = torch.from_numpy(data_dict["bn_conv2"]["scale"]).type(dtype)
	model_params["bn2.bias"] = torch.from_numpy(data_dict["bn_conv2"]["offset"]).type(dtype)

	## Up projection layer 1
	model_params["UpProj1.UpConv1.conv1.weight"] = torch.from_numpy(data_dict["layer2x_br1_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv1.conv1.bias"] = torch.from_numpy(data_dict["layer2x_br1_ConvA"]["bias"].type(dtype)

	model_params["UpProj1.UpConv1.conv2.weight"] = torch.from_numpy(data_dict["layer2x_br1_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv1.conv2.bias"] = torch.from_numpy(data_dict["layer2x_br1_ConvB"]["bias"].type(dtype)

	model_params["UpProj1.UpConv1.conv3.weight"] = torch.from_numpy(data_dict["layer2x_br1_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv1.conv3.bias"] = torch.from_numpy(data_dict["layer2x_br1_ConvC"]["bias"].type(dtype)

	model_params["UpProj1.UpConv1.conv4.weight"] = torch.from_numpy(data_dict["layer2x_br1_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv1.conv4.bias"] = torch.from_numpy(data_dict["layer2x_br1_ConvD"]["bias"].type(dtype)

	model_params["UpProj1.UpConv1.bn1.weight"] = torch.from_numpy(data_dict["layer2x_br1_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv1.bn1.bias"] = torch.from_numpy(data_dict["layer2x_br1_BN"]['offset']).type(dtype)

	#
	model_params["UpProj1.UpConv2.conv1.weight"] = torch.from_numpy(data_dict["layer2x_br2_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv2.conv1.bias"] = torch.from_numpy(data_dict["layer2x_br2_ConvA"]["bias"].type(dtype)

	model_params["UpProj1.UpConv2.conv2.weight"] = torch.from_numpy(data_dict["layer2x_br2_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv2.conv2.bias"] = torch.from_numpy(data_dict["layer2x_br2_ConvB"]["bias"].type(dtype)

	model_params["UpProj1.UpConv2.conv3.weight"] = torch.from_numpy(data_dict["layer2x_br2_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv2.conv3.bias"] = torch.from_numpy(data_dict["layer2x_br2_ConvC"]["bias"].type(dtype)

	model_params["UpProj1.UpConv2.conv4.weight"] = torch.from_numpy(data_dict["layer2x_br2_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv2.conv4.bias"] = torch.from_numpy(data_dict["layer2x_br2_ConvD"]["bias"].type(dtype)

	model_params["UpProj1.UpConv2.bn.weight"] = torch.from_numpy(data_dict["layer2x_br2_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.UpConv2.bn.bias"] = torch.from_numpy(data_dict["layer2x_br2_BN"]['offset']).type(dtype)

	model_params["UpProj1.conv1.weight"] = torch.from_numpy(data_dict["layer2x_Conv"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.conv1.bias"] = torch.from_numpy(data_dict["layer2x_Conv]["bias"].type(dtype)

	#
	model_params["UpProj1.bn.weight"] = torch.from_numpy(data_dict["layer2x_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj1.bn.bias"] = torch.from_numpy(data_dict["layer2x_BN"]['offset']).type(dtype)

	## Up projection layer 2
	model_params["UpProj2.UpConv1.conv1.weight"] = torch.from_numpy(data_dict["layer4x_br1_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv1.conv1.bias"] = torch.from_numpy(data_dict["layer4x_br1_ConvA"]["bias"].type(dtype)

	model_params["UpProj2.UpConv1.conv2.weight"] = torch.from_numpy(data_dict["layer4x_br1_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv1.conv2.bias"] = torch.from_numpy(data_dict["layer4x_br1_ConvB"]["bias"].type(dtype)

	model_params["UpProj2.UpConv1.conv3.weight"] = torch.from_numpy(data_dict["layer4x_br1_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv1.conv3.bias"] = torch.from_numpy(data_dict["layer4x_br1_ConvC"]["bias"].type(dtype)

	model_params["UpProj2.UpConv1.conv4.weight"] = torch.from_numpy(data_dict["layer4x_br1_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv1.conv4.bias"] = torch.from_numpy(data_dict["layer4x_br1_ConvD"]["bias"].type(dtype)

	model_params["UpProj2.UpConv1.bn1.weight"] = torch.from_numpy(data_dict["layer4x_br1_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv1.bn1.bias"] = torch.from_numpy(data_dict["layer4x_br1_BN"]['offset']).type(dtype)

	#
	model_params["UpProj2.UpConv2.conv1.weight"] = torch.from_numpy(data_dict["layer4x_br2_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv2.conv1.bias"] = torch.from_numpy(data_dict["layer4x_br2_ConvA"]["bias"].type(dtype)

	model_params["UpProj2.UpConv2.conv2.weight"] = torch.from_numpy(data_dict["layer4x_br2_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv2.conv2.bias"] = torch.from_numpy(data_dict["layer4x_br2_ConvB"]["bias"].type(dtype)

	model_params["UpProj2.UpConv2.conv3.weight"] = torch.from_numpy(data_dict["layer4x_br2_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv2.conv3.bias"] = torch.from_numpy(data_dict["layer4x_br2_ConvC"]["bias"].type(dtype)

	model_params["UpProj2.UpConv2.conv4.weight"] = torch.from_numpy(data_dict["layer4x_br2_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv2.conv4.bias"] = torch.from_numpy(data_dict["layer4x_br2_ConvD"]["bias"].type(dtype)

	model_params["UpProj2.UpConv2.bn.weight"] = torch.from_numpy(data_dict["layer4x_br2_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.UpConv2.bn.bias"] = torch.from_numpy(data_dict["layer4x_br2_BN"]['offset']).type(dtype)

	model_params["UpProj2.conv1.weight"] = torch.from_numpy(data_dict["layer4x_Conv"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.conv1.bias"] = torch.from_numpy(data_dict["layer4x_Conv]["bias"].type(dtype)

	#
	model_params["UpProj2.bn.weight"] = torch.from_numpy(data_dict["layer4x_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj2.bn.bias"] = torch.from_numpy(data_dict["layer4x_BN"]['offset']).type(dtype)

	## Up projection layer 3
	model_params["UpProj3.UpConv1.conv1.weight"] = torch.from_numpy(data_dict["layer8x_br1_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv1.conv1.bias"] = torch.from_numpy(data_dict["layer8x_br1_ConvA"]["bias"].type(dtype)

	model_params["UpProj3.UpConv1.conv2.weight"] = torch.from_numpy(data_dict["layer8x_br1_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv1.conv2.bias"] = torch.from_numpy(data_dict["layer8x_br1_ConvB"]["bias"].type(dtype)

	model_params["UpProj3.UpConv1.conv3.weight"] = torch.from_numpy(data_dict["layer8x_br1_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv1.conv3.bias"] = torch.from_numpy(data_dict["layer8x_br1_ConvC"]["bias"].type(dtype)

	model_params["UpProj3.UpConv1.conv4.weight"] = torch.from_numpy(data_dict["layer8x_br1_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv1.conv4.bias"] = torch.from_numpy(data_dict["layer8x_br1_ConvD"]["bias"].type(dtype)

	model_params["UpProj3.UpConv1.bn1.weight"] = torch.from_numpy(data_dict["layer8x_br1_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv1.bn1.bias"] = torch.from_numpy(data_dict["layer8x_br1_BN"]['offset']).type(dtype)

	#
	model_params["UpProj3.UpConv2.conv1.weight"] = torch.from_numpy(data_dict["layer8x_br2_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv2.conv1.bias"] = torch.from_numpy(data_dict["layer8x_br2_ConvA"]["bias"].type(dtype)

	model_params["UpProj3.UpConv2.conv2.weight"] = torch.from_numpy(data_dict["layer8x_br2_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv2.conv2.bias"] = torch.from_numpy(data_dict["layer8x_br2_ConvB"]["bias"].type(dtype)

	model_params["UpProj3.UpConv2.conv3.weight"] = torch.from_numpy(data_dict["layer8x_br2_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv2.conv3.bias"] = torch.from_numpy(data_dict["layer8x_br2_ConvC"]["bias"].type(dtype)

	model_params["UpProj3.UpConv2.conv4.weight"] = torch.from_numpy(data_dict["layer8x_br2_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv2.conv4.bias"] = torch.from_numpy(data_dict["layer8x_br2_ConvD"]["bias"].type(dtype)

	model_params["UpProj3.UpConv2.bn.weight"] = torch.from_numpy(data_dict["layer8x_br2_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.UpConv2.bn.bias"] = torch.from_numpy(data_dict["layer8x_br2_BN"]['offset']).type(dtype)

	model_params["UpProj3.conv1.weight"] = torch.from_numpy(data_dict["layer8x_Conv"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.conv1.bias"] = torch.from_numpy(data_dict["layer8x_Conv]["bias"].type(dtype)

	#
	model_params["UpProj3.bn.weight"] = torch.from_numpy(data_dict["layer8x_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj3.bn.bias"] = torch.from_numpy(data_dict["layer8x_BN"]['offset']).type(dtype)

	## Up projection layer 4
	model_params["UpProj4.UpConv1.conv1.weight"] = torch.from_numpy(data_dict["layer16x_br1_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv1.conv1.bias"] = torch.from_numpy(data_dict["layer16x_br1_ConvA"]["bias"].type(dtype)

	model_params["UpProj4.UpConv1.conv2.weight"] = torch.from_numpy(data_dict["layer16x_br1_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv1.conv2.bias"] = torch.from_numpy(data_dict["layer16x_br1_ConvB"]["bias"].type(dtype)

	model_params["UpProj4.UpConv1.conv3.weight"] = torch.from_numpy(data_dict["layer16x_br1_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv1.conv3.bias"] = torch.from_numpy(data_dict["layer16x_br1_ConvC"]["bias"].type(dtype)

	model_params["UpProj4.UpConv1.conv4.weight"] = torch.from_numpy(data_dict["layer16x_br1_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv1.conv4.bias"] = torch.from_numpy(data_dict["layer16x_br1_ConvD"]["bias"].type(dtype)

	model_params["UpProj4.UpConv1.bn1.weight"] = torch.from_numpy(data_dict["layer16x_br1_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv1.bn1.bias"] = torch.from_numpy(data_dict["layer16x_br1_BN"]['offset']).type(dtype)

	#
	model_params["UpProj4.UpConv2.conv1.weight"] = torch.from_numpy(data_dict["layer16x_br2_ConvA"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv2.conv1.bias"] = torch.from_numpy(data_dict["layer16x_br2_ConvA"]["bias"].type(dtype)

	model_params["UpProj4.UpConv2.conv2.weight"] = torch.from_numpy(data_dict["layer16x_br2_ConvB"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv2.conv2.bias"] = torch.from_numpy(data_dict["layer16x_br2_ConvB"]["bias"].type(dtype)

	model_params["UpProj4.UpConv2.conv3.weight"] = torch.from_numpy(data_dict["layer16x_br2_ConvC"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv2.conv3.bias"] = torch.from_numpy(data_dict["layer16x_br2_ConvC"]["bias"].type(dtype)

	model_params["UpProj4.UpConv2.conv4.weight"] = torch.from_numpy(data_dict["layer16x_br2_ConvD"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv2.conv4.bias"] = torch.from_numpy(data_dict["layer16x_br2_ConvD"]["bias"].type(dtype)

	model_params["UpProj4.UpConv2.bn.weight"] = torch.from_numpy(data_dict["layer16x_br2_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.UpConv2.bn.bias"] = torch.from_numpy(data_dict["layer16x_br2_BN"]['offset']).type(dtype)

	model_params["UpProj4.conv1.weight"] = torch.from_numpy(data_dict["layer16x_Conv"]["weights"]).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.conv1.bias"] = torch.from_numpy(data_dict["layer16x_Conv]["bias"].type(dtype)

	#
	model_params["UpProj4.bn.weight"] = torch.from_numpy(data_dict["layer16x_BN"]['scale']).type(dtype).permute(3,2,0,1)
	model_params["UpProj4.bn.bias"] = torch.from_numpy(data_dict["layer16x_BN"]['offset']).type(dtype)

	# final conv layer
	model_params['conv3.weight'] = torch.from_numpy(data_dict['ConvPred']['weight']).type(dtype).permute(3,2,0,1)
	model_params['conv3.bias'] = torch.from_numpy(data_dict['ConvPred']['weight']).type(dtype)

	return model_params
