
import pandas as pd
import re

# Multiline log text as provided by the user
log = """
Epoch [1/20], Step [50], Total Loss: 17.7697, Content Loss: 10.3102, Style Loss: 0.0001, TV Loss: 0.3905
Saved visualization to vis_50.png
Epoch [1/20], Step [100], Total Loss: 14.1965, Content Loss: 8.8741, Style Loss: 0.0000, TV Loss: 0.4033
Saved visualization to vis_100.png
Epoch [1/20], Step [150], Total Loss: 12.7345, Content Loss: 9.2652, Style Loss: 0.0000, TV Loss: 0.4247
Saved visualization to vis_150.png
Epoch [1/20], Step [200], Total Loss: 10.8214, Content Loss: 7.7158, Style Loss: 0.0000, TV Loss: 0.4261
Saved visualization to vis_200.png
Epoch [1/20], Step [250], Total Loss: 11.5721, Content Loss: 8.2206, Style Loss: 0.0000, TV Loss: 0.4212
Saved visualization to vis_250.png
Epoch [1/20], Step [300], Total Loss: 11.1453, Content Loss: 7.9590, Style Loss: 0.0000, TV Loss: 0.4233
Saved visualization to vis_300.png
Epoch [1/20], Step [350], Total Loss: 10.5814, Content Loss: 7.3020, Style Loss: 0.0000, TV Loss: 0.4254
Saved visualization to vis_350.png
Epoch [1/20], Step [400], Total Loss: 10.0408, Content Loss: 7.0122, Style Loss: 0.0000, TV Loss: 0.4159
Saved visualization to vis_400.png
Epoch [1/20], Step [450], Total Loss: 9.7494, Content Loss: 6.8001, Style Loss: 0.0000, TV Loss: 0.4432
Saved visualization to vis_450.png
Epoch [1/20], Step [500], Total Loss: 11.0095, Content Loss: 8.0757, Style Loss: 0.0000, TV Loss: 0.4180
Saved visualization to vis_500.png
Saved checkpoint: checkpoints/transformer_epoch1_step500.pth
Epoch [1/20], Step [550], Total Loss: 9.1989, Content Loss: 6.5513, Style Loss: 0.0000, TV Loss: 0.4345
Saved visualization to vis_550.png
Epoch [1/20], Step [600], Total Loss: 9.9530, Content Loss: 7.1320, Style Loss: 0.0000, TV Loss: 0.4278
Saved visualization to vis_600.png
Epoch [1/20], Step [650], Total Loss: 9.0743, Content Loss: 6.3627, Style Loss: 0.0000, TV Loss: 0.4318
Saved visualization to vis_650.png
Epoch [1/20], Step [700], Total Loss: 9.9116, Content Loss: 7.2762, Style Loss: 0.0000, TV Loss: 0.4166
Saved visualization to vis_700.png
Epoch [1/20], Step [750], Total Loss: 8.8730, Content Loss: 6.3160, Style Loss: 0.0000, TV Loss: 0.4256
Saved visualization to vis_750.png
Epoch [1/20], Step [800], Total Loss: 8.7490, Content Loss: 6.5684, Style Loss: 0.0000, TV Loss: 0.4389
Saved visualization to vis_800.png
Epoch [1/20], Step [850], Total Loss: 8.9036, Content Loss: 6.5001, Style Loss: 0.0000, TV Loss: 0.4171
Saved visualization to vis_850.png
Epoch [1/20], Step [900], Total Loss: 9.2226, Content Loss: 6.6365, Style Loss: 0.0000, TV Loss: 0.4294
Saved visualization to vis_900.png
Epoch [1/20], Step [950], Total Loss: 8.6635, Content Loss: 6.2206, Style Loss: 0.0000, TV Loss: 0.4186
Saved visualization to vis_950.png
Epoch [1/20], Step [1000], Total Loss: 8.9974, Content Loss: 6.6078, Style Loss: 0.0000, TV Loss: 0.4331
Saved visualization to vis_1000.png
Saved checkpoint: checkpoints/transformer_epoch1_step1000.pth
Epoch [1/20], Step [1050], Total Loss: 8.8902, Content Loss: 6.4134, Style Loss: 0.0000, TV Loss: 0.4214
Saved visualization to vis_1050.png
Epoch [1/20], Step [1100], Total Loss: 8.5138, Content Loss: 6.0968, Style Loss: 0.0000, TV Loss: 0.4279
Saved visualization to vis_1100.png
Epoch [1/20], Step [1150], Total Loss: 9.0817, Content Loss: 6.6505, Style Loss: 0.0000, TV Loss: 0.4290
Saved visualization to vis_1150.png
Epoch [1/20], Step [1200], Total Loss: 8.6351, Content Loss: 6.1521, Style Loss: 0.0000, TV Loss: 0.4177
Saved visualization to vis_1200.png
Epoch [1/20], Step [1250], Total Loss: 8.2521, Content Loss: 5.7170, Style Loss: 0.0000, TV Loss: 0.4232
Saved visualization to vis_1250.png
Epoch [1/20], Step [1300], Total Loss: 8.2406, Content Loss: 5.5775, Style Loss: 0.0000, TV Loss: 0.4243
Saved visualization to vis_1300.png
Epoch [1/20], Step [1350], Total Loss: 9.0823, Content Loss: 6.3460, Style Loss: 0.0000, TV Loss: 0.4215
Saved visualization to vis_1350.png
Epoch [1/20], Step [1400], Total Loss: 7.9566, Content Loss: 5.3751, Style Loss: 0.0000, TV Loss: 0.4071
Saved visualization to vis_1400.png
Epoch [1/20], Step [1650], Total Loss: 8.1464, Content Loss: 5.8323, Style Loss: 0.0000, TV Loss: 0.4248
Saved visualization to vis_1650.png
Epoch [1/20], Step [1700], Total Loss: 8.4031, Content Loss: 5.5243, Style Loss: 0.0000, TV Loss: 0.4297
Saved visualization to vis_1700.png
Epoch [1/20], Step [1950], Total Loss: 7.8257, Content Loss: 5.5373, Style Loss: 0.0000, TV Loss: 0.4201
Saved visualization to vis_1950.png
Epoch [1/20], Step [2000], Total Loss: 7.9380, Content Loss: 5.7369, Style Loss: 0.0000, TV Loss: 0.4247
Saved visualization to vis_2000.png
Saved checkpoint: checkpoints/transformer_epoch1_step2000.pth
Epoch [1/20], Step [2050], Total Loss: 8.0222, Content Loss: 5.7604, Style Loss: 0.0000, TV Loss: 0.4261
Saved visualization to vis_2050.png
Epoch [1/20], Step [2100], Total Loss: 7.5957, Content Loss: 5.3654, Style Loss: 0.0000, TV Loss: 0.4262
Saved visualization to vis_2100.png
Epoch [1/20], Step [2150], Total Loss: 7.9371, Content Loss: 5.7068, Style Loss: 0.0000, TV Loss: 0.4305
Saved visualization to vis_2150.png
Epoch [1/20], Step [2200], Total Loss: 8.0651, Content Loss: 5.6491, Style Loss: 0.0000, TV Loss: 0.4396
Saved visualization to vis_2200.png
Epoch [1/20], Step [2250], Total Loss: 7.8628, Content Loss: 5.5415, Style Loss: 0.0000, TV Loss: 0.4259
Saved visualization to vis_2250.png
Epoch [1/20], Step [2300], Total Loss: 7.7988, Content Loss: 5.4219, Style Loss: 0.0000, TV Loss: 0.4383
Saved visualization to vis_2300.png
Epoch [1/20], Step [2350], Total Loss: 7.5598, Content Loss: 5.3784, Style Loss: 0.0000, TV Loss: 0.4308
Saved visualization to vis_2350.png
Epoch [1/20], Step [2400], Total Loss: 8.0873, Content Loss: 5.4723, Style Loss: 0.0000, TV Loss: 0.4168
Saved visualization to vis_2400.png
Epoch [1/20], Step [2450], Total Loss: 7.2121, Content Loss: 4.9675, Style Loss: 0.0000, TV Loss: 0.4309
Saved visualization to vis_2450.png
Epoch [1/20], Step [2500], Total Loss: 7.6449, Content Loss: 5.5196, Style Loss: 0.0000, TV Loss: 0.4372
Saved visualization to vis_2500.png
Saved checkpoint: checkpoints/transformer_epoch1_step2500.pth
Epoch [1/20], Step [2550], Total Loss: 8.0118, Content Loss: 5.8182, Style Loss: 0.0000, TV Loss: 0.4360
Saved visualization to vis_2550.png
Epoch [1/20], Step [2600], Total Loss: 7.5071, Content Loss: 5.0239, Style Loss: 0.0000, TV Loss: 0.4223
Saved visualization to vis_2600.png
Epoch [1/20], Step [2650], Total Loss: 8.4189, Content Loss: 5.9997, Style Loss: 0.0000, TV Loss: 0.4256
Saved visualization to vis_2650.png
Epoch [1/20], Step [2700], Total Loss: 7.4910, Content Loss: 5.3578, Style Loss: 0.0000, TV Loss: 0.4311
Saved visualization to vis_2700.png
Epoch [1/20], Step [2750], Total Loss: 7.2550, Content Loss: 5.1822, Style Loss: 0.0000, TV Loss: 0.4280
Saved visualization to vis_2750.png
Epoch [1/20], Step [2800], Total Loss: 7.2504, Content Loss: 4.9488, Style Loss: 0.0000, TV Loss: 0.4101
Saved visualization to vis_2800.png
Epoch [1/20], Step [2850], Total Loss: 7.5424, Content Loss: 5.2542, Style Loss: 0.0000, TV Loss: 0.4145
Saved visualization to vis_2850.png
Epoch [1/20], Step [2900], Total Loss: 7.6857, Content Loss: 5.1931, Style Loss: 0.0000, TV Loss: 0.4253
Saved visualization to vis_2900.png
Epoch [1/20], Step [2950], Total Loss: 7.1625, Content Loss: 5.0853, Style Loss: 0.0000, TV Loss: 0.4251
Saved visualization to vis_2950.png
Epoch [1/20], Step [3000], Total Loss: 7.2986, Content Loss: 4.9324, Style Loss: 0.0000, TV Loss: 0.4281
Saved visualization to vis_3000.png
Saved checkpoint: checkpoints/transformer_epoch1_step3000.pth
Epoch [1/20], Step [3050], Total Loss: 7.3421, Content Loss: 4.9812, Style Loss: 0.0000, TV Loss: 0.4297
Saved visualization to vis_3050.png
Epoch [1/20], Step [3100], Total Loss: 7.4995, Content Loss: 5.2637, Style Loss: 0.0000, TV Loss: 0.4253
Saved visualization to vis_3100.png
Epoch [1/20], Step [3150], Total Loss: 7.3160, Content Loss: 4.8780, Style Loss: 0.0000, TV Loss: 0.4312
Saved visualization to vis_3150.png
Epoch [1/20], Step [3200], Total Loss: 7.8958, Content Loss: 5.4992, Style Loss: 0.0000, TV Loss: 0.4246
Saved visualization to vis_3200.png
Epoch [1/20], Step [3250], Total Loss: 7.3133, Content Loss: 5.1040, Style Loss: 0.0000, TV Loss: 0.4260
Saved visualization to vis_3250.png
Epoch [1/20], Step [3300], Total Loss: 7.4956, Content Loss: 5.1910, Style Loss: 0.0000, TV Loss: 0.4340
Saved visualization to vis_3300.png
Epoch [1/20], Step [3350], Total Loss: 7.2538, Content Loss: 4.9872, Style Loss: 0.0000, TV Loss: 0.4274
Saved visualization to vis_3350.png
Epoch [1/20], Step [3400], Total Loss: 7.6178, Content Loss: 5.2062, Style Loss: 0.0000, TV Loss: 0.4213
Saved visualization to vis_3400.png
Epoch [1/20], Step [3450], Total Loss: 7.4167, Content Loss: 5.1090, Style Loss: 0.0000, TV Loss: 0.4242
Saved visualization to vis_3450.png
Epoch [1/20], Step [3500], Total Loss: 6.9909, Content Loss: 4.9471, Style Loss: 0.0000, TV Loss: 0.4231
Saved visualization to vis_3500.png
Saved checkpoint: checkpoints/transformer_epoch1_step3500.pth
Epoch [1/20], Step [3550], Total Loss: 7.3114, Content Loss: 4.9164, Style Loss: 0.0000, TV Loss: 0.4231
Saved visualization to vis_3550.png
Epoch [1/20], Step [3600], Total Loss: 7.9575, Content Loss: 5.2199, Style Loss: 0.0000, TV Loss: 0.4225
Saved visualization to vis_3600.png
Epoch [1/20], Step [3650], Total Loss: 7.9577, Content Loss: 5.5869, Style Loss: 0.0000, TV Loss: 0.4289
Saved visualization to vis_3650.png
Epoch [1/20], Step [3700], Total Loss: 7.1331, Content Loss: 4.9459, Style Loss: 0.0000, TV Loss: 0.4302
Saved visualization to vis_3700.png
Epoch [1/20], Step [3750], Total Loss: 7.0059, Content Loss: 4.8088, Style Loss: 0.0000, TV Loss: 0.4305
Saved visualization to vis_3750.png
Epoch [1/20], Step [3800], Total Loss: 7.1647, Content Loss: 5.0130, Style Loss: 0.0000, TV Loss: 0.4272
Saved visualization to vis_3800.png
Epoch [1/20], Step [3850], Total Loss: 7.2978, Content Loss: 5.0888, Style Loss: 0.0000, TV Loss: 0.4247
Saved visualization to vis_3850.png
Epoch [1/20], Step [3900], Total Loss: 6.9440, Content Loss: 4.9053, Style Loss: 0.0000, TV Loss: 0.4211
Saved visualization to vis_3900.png
Epoch [1/20], Step [3950], Total Loss: 7.1414, Content Loss: 4.9631, Style Loss: 0.0000, TV Loss: 0.4337
Saved visualization to vis_3950.png
Epoch [1/20], Step [4250], Total Loss: 6.7976, Content Loss: 4.6220, Style Loss: 0.0000, TV Loss: 0.4232
Saved visualization to vis_4250.png
Epoch [1/20], Step [4300], Total Loss: 7.2308, Content Loss: 5.0722, Style Loss: 0.0000, TV Loss: 0.4310
Saved visualization to vis_4300.png
Epoch [1/20], Step [4350], Total Loss: 7.6494, Content Loss: 4.9209, Style Loss: 0.0000, TV Loss: 0.4133
Saved visualization to vis_4350.png
Epoch [1/20], Step [4400], Total Loss: 6.7826, Content Loss: 4.4632, Style Loss: 0.0000, TV Loss: 0.4290
Saved visualization to vis_4400.png
Epoch [1/20], Step [4450], Total Loss: 7.0736, Content Loss: 4.9451, Style Loss: 0.0000, TV Loss: 0.4263
Saved visualization to vis_4450.png
Epoch [1/20], Step [4500], Total Loss: 7.9454, Content Loss: 5.6765, Style Loss: 0.0000, TV Loss: 0.4265
Saved visualization to vis_4500.png
Saved checkpoint: checkpoints/transformer_epoch1_step4500.pth
Epoch [1/20], Step [4550], Total Loss: 6.8768, Content Loss: 4.6417, Style Loss: 0.0000, TV Loss: 0.4220
Saved visualization to vis_4550.png
Epoch [1/20], Step [4600], Total Loss: 7.3271, Content Loss: 4.9917, Style Loss: 0.0000, TV Loss: 0.4169
Saved visualization to vis_4600.png
Epoch [1/20], Step [4650], Total Loss: 7.1994, Content Loss: 4.9935, Style Loss: 0.0000, TV Loss: 0.4197
Saved visualization to vis_4650.png
Epoch [1/20], Step [4700], Total Loss: 7.3528, Content Loss: 4.9474, Style Loss: 0.0000, TV Loss: 0.4345
Saved visualization to vis_4700.png
Epoch [1/20], Step [4750], Total Loss: 6.7437, Content Loss: 4.6262, Style Loss: 0.0000, TV Loss: 0.4253
Saved visualization to vis_4750.png
Epoch [1/20], Step [4800], Total Loss: 7.2854, Content Loss: 5.1884, Style Loss: 0.0000, TV Loss: 0.4250
Saved visualization to vis_4800.png
Epoch [1/20], Step [4850], Total Loss: 7.0358, Content Loss: 4.6397, Style Loss: 0.0000, TV Loss: 0.4231
Saved visualization to vis_4850.png
Epoch [1/20], Step [4900], Total Loss: 6.9618, Content Loss: 4.7608, Style Loss: 0.0000, TV Loss: 0.4292
Saved visualization to vis_4900.png
Epoch [1/20], Step [4950], Total Loss: 6.6496, Content Loss: 4.5754, Style Loss: 0.0000, TV Loss: 0.4223
Saved visualization to vis_4950.png
Epoch [1/20], Step [5000], Total Loss: 7.0273, Content Loss: 4.7943, Style Loss: 0.0000, TV Loss: 0.4174
Saved visualization to vis_5000.png
Saved checkpoint: checkpoints/transformer_epoch1_step5000.pth
Epoch [1/20], Step [5050], Total Loss: 7.0065, Content Loss: 4.5519, Style Loss: 0.0000, TV Loss: 0.4278
Saved visualization to vis_5050.png
Epoch [1/20], Step [5100], Total Loss: 6.5491, Content Loss: 4.4814, Style Loss: 0.0000, TV Loss: 0.4267
Saved visualization to vis_5100.png
Epoch [1/20], Step [5150], Total Loss: 6.6640, Content Loss: 4.4633, Style Loss: 0.0000, TV Loss: 0.4196
Saved visualization to vis_5150.png
Epoch [1/20], Step [5200], Total Loss: 7.6577, Content Loss: 5.4899, Style Loss: 0.0000, TV Loss: 0.4194
Saved visualization to vis_5200.png
Epoch [1/20], Step [5250], Total Loss: 6.8988, Content Loss: 4.7619, Style Loss: 0.0000, TV Loss: 0.4310
Saved visualization to vis_5250.png
Epoch [1/20], Step [5300], Total Loss: 6.5451, Content Loss: 4.4676, Style Loss: 0.0000, TV Loss: 0.4288
Saved visualization to vis_5300.png
Epoch [1/20], Step [5350], Total Loss: 6.4668, Content Loss: 4.4868, Style Loss: 0.0000, TV Loss: 0.4284
Saved visualization to vis_5350.png
Epoch [1/20], Step [5400], Total Loss: 6.9623, Content Loss: 4.7041, Style Loss: 0.0000, TV Loss: 0.4242
Saved visualization to vis_5400.png
Epoch [1/20], Step [5450], Total Loss: 6.5436, Content Loss: 4.3857, Style Loss: 0.0000, TV Loss: 0.4313
Saved visualization to vis_5450.png
Epoch [1/20], Step [5500], Total Loss: 7.6399, Content Loss: 4.9512, Style Loss: 0.0000, TV Loss: 0.4089
Saved visualization to vis_5500.png
Saved checkpoint: checkpoints/transformer_epoch1_step5500.pth
Epoch [1/20], Step [5550], Total Loss: 6.6341, Content Loss: 4.5044, Style Loss: 0.0000, TV Loss: 0.4238
Saved visualization to vis_5550.png
Epoch [1/20], Step [5600], Total Loss: 6.8110, Content Loss: 4.4282, Style Loss: 0.0000, TV Loss: 0.4203
Saved visualization to vis_5600.png
Epoch [1/20], Step [5650], Total Loss: 6.7720, Content Loss: 4.4172, Style Loss: 0.0000, TV Loss: 0.4232
Saved visualization to vis_5650.png
Epoch [1/20], Step [5700], Total Loss: 7.0387, Content Loss: 4.7499, Style Loss: 0.0000, TV Loss: 0.4202
Saved visualization to vis_5700.png
Epoch [1/20], Step [5750], Total Loss: 6.6748, Content Loss: 4.6195, Style Loss: 0.0000, TV Loss: 0.4221
Saved visualization to vis_5750.png
Epoch [1/20], Step [5800], Total Loss: 6.8483, Content Loss: 4.6896, Style Loss: 0.0000, TV Loss: 0.4237
Saved visualization to vis_5800.png
Epoch [1/20], Step [5850], Total Loss: 6.6236, Content Loss: 4.4469, Style Loss: 0.0000, TV Loss: 0.4231
Saved visualization to vis_5850.png
Epoch [1/20], Step [5900], Total Loss: 6.8545, Content Loss: 4.6853, Style Loss: 0.0000, TV Loss: 0.4279
Saved visualization to vis_5900.png
Epoch [1/20], Step [5950], Total Loss: 6.9628, Content Loss: 4.4432, Style Loss: 0.0000, TV Loss: 0.4197
Saved visualization to vis_5950.png
Epoch [1/20], Step [6000], Total Loss: 7.1811, Content Loss: 4.8335, Style Loss: 0.0000, TV Loss: 0.4323
Saved visualization to vis_6000.png
Saved checkpoint: checkpoints/transformer_epoch1_step6000.pth
Epoch [1/20], Step [6050], Total Loss: 7.1003, Content Loss: 5.0339, Style Loss: 0.0000, TV Loss: 0.4201
Saved visualization to vis_6050.png
Epoch [1/20], Step [6100], Total Loss: 7.0571, Content Loss: 4.9198, Style Loss: 0.0000, TV Loss: 0.4119
Saved visualization to vis_6100.png
Epoch [1/20], Step [6150], Total Loss: 6.6255, Content Loss: 4.3218, Style Loss: 0.0000, TV Loss: 0.4352
Saved visualization to vis_6150.png
Epoch [1/20], Step [6200], Total Loss: 6.9025, Content Loss: 4.7004, Style Loss: 0.0000, TV Loss: 0.4271
Saved visualization to vis_6200.png
Epoch [1/20], Step [6250], Total Loss: 6.9012, Content Loss: 4.3621, Style Loss: 0.0000, TV Loss: 0.4261
Saved visualization to vis_6250.png
Epoch [1/20], Step [6300], Total Loss: 7.2408, Content Loss: 4.9668, Style Loss: 0.0000, TV Loss: 0.4178
Saved visualization to vis_6300.png
Epoch [1/20], Step [6550], Total Loss: 6.5681, Content Loss: 4.6144, Style Loss: 0.0000, TV Loss: 0.4295
Saved visualization to vis_6550.png
Epoch [1/20], Step [6600], Total Loss: 7.3612, Content Loss: 5.0815, Style Loss: 0.0000, TV Loss: 0.4163
Saved visualization to vis_6600.png
Epoch [1/20], Step [6650], Total Loss: 6.8065, Content Loss: 4.5840, Style Loss: 0.0000, TV Loss: 0.4275
Saved visualization to vis_6650.png
Epoch [1/20], Step [6700], Total Loss: 6.6473, Content Loss: 4.4843, Style Loss: 0.0000, TV Loss: 0.4220
Saved visualization to vis_6700.png
Epoch [1/20], Step [6750], Total Loss: 6.9274, Content Loss: 4.6992, Style Loss: 0.0000, TV Loss: 0.4323
Saved visualization to vis_6750.png
Epoch [1/20], Step [6800], Total Loss: 6.5717, Content Loss: 4.0506, Style Loss: 0.0000, TV Loss: 0.4211
Saved visualization to vis_6800.png
Epoch [1/20], Step [6850], Total Loss: 6.4434, Content Loss: 4.2956, Style Loss: 0.0000, TV Loss: 0.4191
Saved visualization to vis_6850.png
Epoch [1/20], Step [6900], Total Loss: 6.8081, Content Loss: 4.7394, Style Loss: 0.0000, TV Loss: 0.4338
Saved visualization to vis_6900.png
Epoch [1/20], Step [6950], Total Loss: 7.2177, Content Loss: 4.9833, Style Loss: 0.0000, TV Loss: 0.4293
Saved visualization to vis_6950.png
Epoch [1/20], Step [7000], Total Loss: 6.3946, Content Loss: 4.4619, Style Loss: 0.0000, TV Loss: 0.4374
Saved visualization to vis_7000.png
Saved checkpoint: checkpoints/transformer_epoch1_step7000.pth
Epoch [1/20], Step [7050], Total Loss: 7.3903, Content Loss: 5.2265, Style Loss: 0.0000, TV Loss: 0.4290
Saved visualization to vis_7050.png
Epoch [1/20], Step [7100], Total Loss: 6.4765, Content Loss: 4.5039, Style Loss: 0.0000, TV Loss: 0.4244
Saved visualization to vis_7100.png
Epoch [1/20], Step [7150], Total Loss: 6.5024, Content Loss: 4.4826, Style Loss: 0.0000, TV Loss: 0.4269
Saved visualization to vis_7150.png
Epoch [1/20], Step [7200], Total Loss: 6.6869, Content Loss: 4.2396, Style Loss: 0.0000, TV Loss: 0.4197
Saved visualization to vis_7200.png
Epoch [1/20], Step [7250], Total Loss: 6.5899, Content Loss: 4.5312, Style Loss: 0.0000, TV Loss: 0.4215
Saved visualization to vis_7250.png
Epoch [1/20], Step [7300], Total Loss: 6.7399, Content Loss: 4.6213, Style Loss: 0.0000, TV Loss: 0.4256
Saved visualization to vis_7300.png
Epoch [1/20], Step [7350], Total Loss: 7.3440, Content Loss: 4.8617, Style Loss: 0.0000, TV Loss: 0.4331
Saved visualization to vis_7350.png
Epoch [1/20], Step [7400], Total Loss: 6.8976, Content Loss: 4.7158, Style Loss: 0.0000, TV Loss: 0.4319
Saved visualization to vis_7400.png
Epoch [1/20], Step [7450], Total Loss: 7.0155, Content Loss: 4.8347, Style Loss: 0.0000, TV Loss: 0.4328
Saved visualization to vis_7450.png
Epoch [1/20], Step [7500], Total Loss: 7.0719, Content Loss: 4.5593, Style Loss: 0.0000, TV Loss: 0.4293
Saved visualization to vis_7500.png
Saved checkpoint: checkpoints/transformer_epoch1_step7500.pth
Epoch [1/20], Step [7550], Total Loss: 6.7126, Content Loss: 4.5731, Style Loss: 0.0000, TV Loss: 0.4243
Saved visualization to vis_7550.png
Epoch [1/20], Step [7600], Total Loss: 7.3321, Content Loss: 4.4399, Style Loss: 0.0000, TV Loss: 0.4075
Saved visualization to vis_7600.png
Epoch [1/20], Step [7650], Total Loss: 6.7622, Content Loss: 4.3708, Style Loss: 0.0000, TV Loss: 0.4213
Saved visualization to vis_7650.png
Epoch [1/20], Step [7700], Total Loss: 6.5682, Content Loss: 4.1862, Style Loss: 0.0000, TV Loss: 0.4330
Saved visualization to vis_7700.png
Epoch [1/20], Step [7750], Total Loss: 6.4073, Content Loss: 4.3953, Style Loss: 0.0000, TV Loss: 0.4272
Saved visualization to vis_7750.png
Epoch [1/20], Step [7800], Total Loss: 6.8102, Content Loss: 4.5724, Style Loss: 0.0000, TV Loss: 0.4204
Saved visualization to vis_7800.png
Epoch [1/20], Step [7850], Total Loss: 6.6713, Content Loss: 4.3836, Style Loss: 0.0000, TV Loss: 0.4368
Saved visualization to vis_7850.png
Epoch [1/20], Step [7900], Total Loss: 6.5171, Content Loss: 4.3608, Style Loss: 0.0000, TV Loss: 0.4266
Saved visualization to vis_7900.png
Epoch [1/20], Step [7950], Total Loss: 6.5823, Content Loss: 4.4988, Style Loss: 0.0000, TV Loss: 0.4323
Saved visualization to vis_7950.png
Epoch [1/20], Step [8000], Total Loss: 6.5318, Content Loss: 4.2514, Style Loss: 0.0000, TV Loss: 0.4196
Saved visualization to vis_8000.png
Saved checkpoint: checkpoints/transformer_epoch1_step8000.pth
Epoch [1/20], Step [8050], Total Loss: 6.8163, Content Loss: 4.6747, Style Loss: 0.0000, TV Loss: 0.4247
Saved visualization to vis_8050.png
Epoch [1/20], Step [8100], Total Loss: 6.6084, Content Loss: 4.4639, Style Loss: 0.0000, TV Loss: 0.4259
Saved visualization to vis_8100.png
Epoch [1/20], Step [8150], Total Loss: 6.2423, Content Loss: 4.2203, Style Loss: 0.0000, TV Loss: 0.4243
Saved visualization to vis_8150.png
Epoch [1/20], Step [8200], Total Loss: 7.2114, Content Loss: 4.9746, Style Loss: 0.0000, TV Loss: 0.4264
Saved visualization to vis_8200.png
Epoch [1/20], Step [8250], Total Loss: 6.6199, Content Loss: 4.5864, Style Loss: 0.0000, TV Loss: 0.4270
Saved visualization to vis_8250.png
Epoch [1/20], Step [8300], Total Loss: 6.4823, Content Loss: 4.2023, Style Loss: 0.0000, TV Loss: 0.4190
Saved visualization to vis_8300.png
Epoch [1/20], Step [8350], Total Loss: 6.6114, Content Loss: 4.5624, Style Loss: 0.0000, TV Loss: 0.4280
Saved visualization to vis_8350.png
Epoch [1/20], Step [8400], Total Loss: 6.9633, Content Loss: 4.6726, Style Loss: 0.0000, TV Loss: 0.4270
Saved visualization to vis_8400.png
Epoch [1/20], Step [8450], Total Loss: 6.4933, Content Loss: 4.2894, Style Loss: 0.0000, TV Loss: 0.4266
Saved visualization to vis_8450.png
Epoch [1/20], Step [8500], Total Loss: 6.6949, Content Loss: 4.5884, Style Loss: 0.0000, TV Loss: 0.4222
Saved visualization to vis_8500.png
Saved checkpoint: checkpoints/transformer_epoch1_step8500.pth
Epoch [1/20], Step [8550], Total Loss: 6.7534, Content Loss: 4.3424, Style Loss: 0.0000, TV Loss: 0.4232
Saved visualization to vis_8550.png
Epoch [1/20], Step [8600], Total Loss: 6.6064, Content Loss: 4.4256, Style Loss: 0.0000, TV Loss: 0.4213
Saved visualization to vis_8600.png
Epoch [1/20], Step [8650], Total Loss: 7.8356, Content Loss: 5.2413, Style Loss: 0.0000, TV Loss: 0.4156
Saved visualization to vis_8650.png
Epoch [1/20], Step [8700], Total Loss: 6.3199, Content Loss: 4.3276, Style Loss: 0.0000, TV Loss: 0.4246
Saved visualization to vis_8700.png
Epoch [1/20], Step [8750], Total Loss: 6.3988, Content Loss: 4.1504, Style Loss: 0.0000, TV Loss: 0.4300
Saved visualization to vis_8750.png
Epoch [1/20], Step [8800], Total Loss: 7.2969, Content Loss: 5.1553, Style Loss: 0.0000, TV Loss: 0.4268
Saved visualization to vis_8800.png
Epoch [1/20], Step [8850], Total Loss: 6.2799, Content Loss: 4.0602, Style Loss: 0.0000, TV Loss: 0.4258
Saved visualization to vis_8850.png
Epoch [1/20], Step [8900], Total Loss: 6.6523, Content Loss: 4.3796, Style Loss: 0.0000, TV Loss: 0.4223
Saved visualization to vis_8900.png
Epoch [1/20], Step [8950], Total Loss: 6.6983, Content Loss: 4.5725, Style Loss: 0.0000, TV Loss: 0.4242
Saved visualization to vis_8950.png
Epoch [1/20], Step [9000], Total Loss: 7.2162, Content Loss: 4.9407, Style Loss: 0.0000, TV Loss: 0.4263
Saved visualization to vis_9000.png
Saved checkpoint: checkpoints/transformer_epoch1_step9000.pth
Epoch [1/20], Step [9050], Total Loss: 6.8462, Content Loss: 4.4282, Style Loss: 0.0000, TV Loss: 0.4240
Saved visualization to vis_9050.png
Epoch [1/20], Step [9100], Total Loss: 6.4698, Content Loss: 4.5832, Style Loss: 0.0000, TV Loss: 0.4279
Saved visualization to vis_9100.png
Epoch [1/20], Step [9150], Total Loss: 6.4274, Content Loss: 4.5342, Style Loss: 0.0000, TV Loss: 0.4273
Saved visualization to vis_9150.png
Epoch [1/20], Step [9200], Total Loss: 6.3341, Content Loss: 4.2472, Style Loss: 0.0000, TV Loss: 0.4290
Saved visualization to vis_9200.png
Epoch [1/20], Step [9250], Total Loss: 6.8112, Content Loss: 4.4325, Style Loss: 0.0000, TV Loss: 0.4155
Saved visualization to vis_9250.png
Epoch [1/20], Step [9300], Total Loss: 6.5027, Content Loss: 4.4770, Style Loss: 0.0000, TV Loss: 0.4278
Saved visualization to vis_9300.png
Epoch [1/20], Step [9350], Total Loss: 6.7912, Content Loss: 4.4814, Style Loss: 0.0000, TV Loss: 0.4245
Saved visualization to vis_9350.png
Epoch [1/20], Step [9400], Total Loss: 6.6626, Content Loss: 4.4945, Style Loss: 0.0000, TV Loss: 0.4154
Saved visualization to vis_9400.png
Epoch [1/20], Step [9450], Total Loss: 6.5642, Content Loss: 4.3914, Style Loss: 0.0000, TV Loss: 0.4327
Saved visualization to vis_9450.png
Epoch [1/20], Step [9500], Total Loss: 6.8761, Content Loss: 4.7672, Style Loss: 0.0000, TV Loss: 0.4234
Saved visualization to vis_9500.png
Saved checkpoint: checkpoints/transformer_epoch1_step9500.pth
Epoch [1/20], Step [9550], Total Loss: 6.6238, Content Loss: 4.6313, Style Loss: 0.0000, TV Loss: 0.4306
Saved visualization to vis_9550.png
Epoch [1/20], Step [9600], Total Loss: 6.1959, Content Loss: 4.2354, Style Loss: 0.0000, TV Loss: 0.4262
Saved visualization to vis_9600.png
Epoch [1/20], Step [9650], Total Loss: 6.6921, Content Loss: 4.7619, Style Loss: 0.0000, TV Loss: 0.4329
Saved visualization to vis_9650.png
Epoch [1/20], Step [9700], Total Loss: 6.5849, Content Loss: 4.4923, Style Loss: 0.0000, TV Loss: 0.4254
Saved visualization to vis_9700.png
Epoch [1/20], Step [9750], Total Loss: 6.4599, Content Loss: 4.3444, Style Loss: 0.0000, TV Loss: 0.4222
Saved visualization to vis_9750.png
Epoch [1/20], Step [9800], Total Loss: 6.7093, Content Loss: 4.3759, Style Loss: 0.0000, TV Loss: 0.4233
Saved visualization to vis_9800.png
Epoch [1/20], Step [9850], Total Loss: 6.6499, Content Loss: 4.3681, Style Loss: 0.0000, TV Loss: 0.4249
Saved visualization to vis_9850.png
Epoch [1/20], Step [9900], Total Loss: 7.2320, Content Loss: 4.8934, Style Loss: 0.0000, TV Loss: 0.4290
Saved visualization to vis_9900.png
Epoch [1/20], Step [9950], Total Loss: 7.1469, Content Loss: 4.8682, Style Loss: 0.0000, TV Loss: 0.4293
Saved visualization to vis_9950.png
Epoch [1/20], Step [10000], Total Loss: 6.7890, Content Loss: 4.5399, Style Loss: 0.0000, TV Loss: 0.4258
Saved visualization to vis_10000.png
Saved checkpoint: checkpoints/transformer_epoch1_step10000.pth
Epoch [1/20], Step [10050], Total Loss: 6.6601, Content Loss: 4.3951, Style Loss: 0.0000, TV Loss: 0.4312
Saved visualization to vis_10050.png
Epoch [1/20], Step [10100], Total Loss: 6.1987, Content Loss: 4.1388, Style Loss: 0.0000, TV Loss: 0.4285
Saved visualization to vis_10100.png
Epoch [1/20], Step [10150], Total Loss: 6.6689, Content Loss: 4.5166, Style Loss: 0.0000, TV Loss: 0.4310
Saved visualization to vis_10150.png
Epoch [1/20], Step [10200], Total Loss: 6.5824, Content Loss: 4.4639, Style Loss: 0.0000, TV Loss: 0.4306
Saved visualization to vis_10200.png
Epoch [1/20], Step [10250], Total Loss: 6.3230, Content Loss: 4.3323, Style Loss: 0.0000, TV Loss: 0.4255
Saved visualization to vis_10250.png
Epoch [1/20], Step [10300], Total Loss: 6.0677, Content Loss: 4.1607, Style Loss: 0.0000, TV Loss: 0.4353
Saved visualization to vis_10300.png
Epoch [2/20], Step [10350], Total Loss: 6.5130, Content Loss: 4.4669, Style Loss: 0.0000, TV Loss: 0.4214
Saved visualization to vis_10350.png
Epoch [2/20], Step [10400], Total Loss: 6.8049, Content Loss: 4.6614, Style Loss: 0.0000, TV Loss: 0.4329
Saved visualization to vis_10400.png
Epoch [2/20], Step [10450], Total Loss: 6.3581, Content Loss: 4.2589, Style Loss: 0.0000, TV Loss: 0.4312
Saved visualization to vis_10450.png
Epoch [2/20], Step [10500], Total Loss: 6.5731, Content Loss: 4.6278, Style Loss: 0.0000, TV Loss: 0.4309
Saved visualization to vis_10500.png
Saved checkpoint: checkpoints/transformer_epoch2_step10500.pth
Epoch [2/20], Step [10550], Total Loss: 6.6993, Content Loss: 4.6148, Style Loss: 0.0000, TV Loss: 0.4243
Saved visualization to vis_10550.png
Epoch [2/20], Step [10600], Total Loss: 6.3993, Content Loss: 4.3866, Style Loss: 0.0000, TV Loss: 0.4321
Saved visualization to vis_10600.png
"""

# Parse the metrics and related visualization/checkpoint info
rows = []
lines = log.strip().splitlines()
for i, line in enumerate(lines):
    m = re.match(r"Epoch \[(\d+)/(\d+)\], Step \[(\d+)\], Total Loss: ([\d\.]+), Content Loss: ([\d\.]+), Style Loss: ([\d\.]+), TV Loss: ([\d\.]+)", line)
    if m:
        epoch, total_epochs, step, total_loss, content_loss, style_loss, tv_loss = m.groups()
        # Next line: visualization
        vis = ""
        if i + 1 < len(lines):
            m2 = re.match(r"Saved visualization to (.+)", lines[i + 1])
            if m2:
                vis = m2.group(1)
        # Find corresponding checkpoint if exists
        checkpoint = ""
        pattern = f"transformer_epoch{epoch}_step{step}"
        for l in lines:
            if pattern in l and 'Saved checkpoint' in l:
                checkpoint = l.split("Saved checkpoint: ")[1]
                break
        rows.append({
            'epoch': int(epoch),
            'total_epochs': int(total_epochs),
            'step': int(step),
            'total_loss': float(total_loss),
            'content_loss': float(content_loss),
            'style_loss': float(style_loss),
            'tv_loss': float(tv_loss),
            'vis_image': vis,
            'checkpoint': checkpoint
        })

# Create DataFrame and save CSV
df = pd.DataFrame(rows)
csv_path = 'style_transfer_log.csv'
df.to_csv(csv_path, index=False)

# Display to user


# Provide link for download
print(f"[Download structured CSV file](sandbox:{csv_path})")
