from models import (DLD_model, resnet, DenseNet, MobileNet)

#from opts import parse_opts

def getmodel(cnn_name, model_depth, n_classes, in_channels, sample_size):
 
    # ResNet
    if cnn_name == 'resnet':
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        model = resnet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)
        
    elif cnn_name == 'DLD_model':
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        model = DLD_model.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)
   
    # DenseNet
    elif cnn_name == 'DenseNet':
        """
        3D resnet
        model_depth = [121, 169, 201]
        """
        model = DenseNet.generate_model(
            model_depth=model_depth,
            num_classes=n_classes,
            n_input_channels=in_channels)


    # MobileNet
    elif cnn_name == 'MobileNet':
        """
        MobileNet
        "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" 
        """
        model = MobileNet.get_model(
            sample_size=sample_size,
            num_classes=n_classes,
            in_channels=in_channels)
    
    return model