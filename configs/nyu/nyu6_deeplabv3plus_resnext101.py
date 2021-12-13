import cv2

# 1. configuration for inference
nclasses = 6  # 40 or 13
ignore_label = 255
#   official_origin: official origin,
#   blank_crop: croped blank padding,
#   official_crop: [h_range=(45, 471), w_range=(41, 601)] -> (427, 561): official cropping to get best depth ,
#   depth_pred_crop: 640*512 -> dowmsample(320, 240) -> crop(304, 228) -> upsample(640, 480)
data_crop_types = {'official_origin': dict(type='official_origin', padding_size=(512, 640)),
                   'official_origin_val': dict(type='official_origin_val', padding_size=(1024, 1280)),
                   'blank_crop': dict(type='blank_crop', center_crop_size=(512, 640), padding_size=(512, 640)),
                   'official_crop': dict(type='official_crop', h_range=(0, 512), w_range=(0, 640),
                                         padding_size=(512, 640)),
                   'depth_pred_crop': dict(type='depth_pred_crop', downsample=(512, 640), center_crop_size=(510, 638),
                                           upsample=(512, 640), padding_size=(512, 640))}

crop_paras = data_crop_types['official_origin']
crop_paras_val = data_crop_types['official_origin_val']
size_h, size_w = crop_paras['padding_size']
batch_size_per_gpu = 8
batch_size_per_gpu_val = 4
data_channels = ['rgb', 'depth']  # ['rgb', 'hha', 'depth']
image_pad_value = ()
norm_mean = ()
norm_std = ()
if 'rgb' in data_channels:
    image_pad_value += (123.675, 116.280, 103.530)  # when using pre-trained models (ImageNet mean)
    # norm_mean += (0.0, 0.0, 0.0)
    # norm_std += (1.0, 1.0, 1.0)
    norm_mean += (0.485, 0.456, 0.406)
    norm_std += (0.229, 0.224, 0.225)
if 'hha' in data_channels:
    image_pad_value += (123.675, 116.280, 103.530)
    # norm_mean += (0.0, 0.0, 0.0)
    # norm_std += (1.0, 1.0, 1.0)
    norm_mean += (0.485, 0.456, 0.406)
    norm_std += (0.229, 0.224, 0.225)
if 'depth' in data_channels:
    image_pad_value += (0.0,)
    norm_mean += (0.0,)
    norm_std += (1.0,)

# img_norm_cfg = dict(mean=norm_mean,
#                     std=norm_std,
#                     max_pixel_value=255.0)
conv_cfg = dict(type='ShapeConv')  # Conv, ShapeConv
norm_cfg = dict(type='BN')  # 'FRN', 'BN', 'SyncBN', 'GN'
act_cfg = dict(type='Relu', inplace=True)  # Relu, Tlu
multi_label = False

inference = dict(
    gpu_id='0',
    multi_label=multi_label,
    transforms=[
        dict(type='ToTensor'),
    ],
    model=dict(
        # model/encoder
        encoder=dict(
            backbone=dict(
                type='ResNet',
                arch='resnext101_32x8d',  # resnext101_32x8d, resnext50_32x4d, resnet152, resnet101, resnet50
                replace_stride_with_dilation=[False, False, True],
                multi_grid=[1, 2, 4],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                input_type=data_channels
            ),
            enhance=dict(
                type='ASPP',
                from_layer='c5',
                to_layer='enhance',
                in_channels=2048,
                out_channels=256,
                atrous_rates=[6, 12, 18],
                mode='bilinear',
                align_corners=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dropout=0.1,
            ),
        ),
        # model/decoder
        decoder=dict(
            type='GFPN',
            # model/decoder/blocks
            neck=[
                # model/decoder/blocks/block1
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='enhance',
                        adapt_upsample=True,
                    ),
                    lateral=dict(
                        from_layer='c2',
                        type='ConvModule',
                        in_channels=256,
                        out_channels=48,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                    post=None,
                    to_layer='p5',
                ),  # 4
            ],
        ),
        # model/head
        head=dict(
            type='Head',
            in_channels=304,
            inter_channels=256,
            out_channels=nclasses,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            num_convs=2,
            upsample=dict(
                type='Upsample',
                size=(512, 640),
                mode='bilinear',
                align_corners=True,
            ),
        )
    )
)

# 2. configuration for train/test
root_workdir = "/mnt/HDD_4T/nyu_v2/output/"
dataset_type = 'NYUV2Dataset'
dataset_root = '/mnt/HDD_4T/nyu_v2'

common = dict(
    seed=0,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metrics=[
        dict(type='IoU', num_classes=nclasses),
        dict(type='MIoU', num_classes=nclasses, average='equal'),
        dict(type='MIoU', num_classes=nclasses, average='frequency_weighted'),
        dict(type='Accuracy', num_classes=nclasses, average='pixel'),
        dict(type='Accuracy', num_classes=nclasses, average='class'),
    ],
    dist_params=dict(backend='nccl')
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            classes=nclasses,
            crop_paras=crop_paras,
            imglist_name='test.txt',
            channels=data_channels,
            multi_label=multi_label,
        ),
        transforms=inference['transforms'],
        sampler=dict(
            type='DefaultSampler',
        ),
        dataloader=dict(
            type='DataLoader',
            samples_per_gpu=batch_size_per_gpu,
            workers_per_gpu=2,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    ),
    tta=dict(
        scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        biases=[None, None, None, None, None, None],  # bias may change the size ratio
        flip=True,
    ),
    # save_pred=True,
)

## 2.2 configuration for train
max_epochs = 20

train = dict(
    data=dict(
        train=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                classes=nclasses,
                crop_paras=crop_paras_val,
                imglist_name='train_val4.txt',
                channels=data_channels,
                multi_label=multi_label,
            ),
            transforms_rgb=[
                dict(type='RandomBrightnessContrast'),
            ],
            transforms=[
                dict(type='HorizontalFlip', p=0.5),
                dict(type='ToTensor'),
            ],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=batch_size_per_gpu,
                workers_per_gpu=2,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        val=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                classes=nclasses,
                crop_paras=crop_paras,
                imglist_name='val_no4.txt',
                channels=data_channels,
                multi_label=multi_label,
            ),
            transforms_rgb=None,
            transforms=[
                dict(type='ToTensor'),
            ],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=batch_size_per_gpu_val,
                workers_per_gpu=2,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            ),
        ),
    ),
    resume=None,
    criterion=dict(type='GDL_CrossEntropy', ignore_label=ignore_label),
    optimizer=dict(type='Adam', lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001),
    lr_scheduler=dict(type='CosinusLR', warm_up=2, power=0.00001, max_epochs=max_epochs, end_lr=10),
    max_epochs=max_epochs,
    trainval_ratio=1,
    log_interval=10,
    snapshot_interval=1,
    save_best=False,
)
