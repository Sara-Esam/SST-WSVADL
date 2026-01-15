"""
Evaluation functions for SST-WSVADL
"""
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")


def test(net, config, wind, test_loader, test_info, step, model_file=None, 
         test_file=None, i3d=False, rgb_thermal_fusion=False):
    """
    Test function for UR-DMU model evaluation
    """
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        if test_file is None:
            if i3d:
                test_file = f'frame_label/{config.dataset}_gt_i3d.npy'
            else:
                test_file = f'frame_label/{config.dataset}_gt.npy'
        
        frame_gt = np.load(test_file)

        frame_predict = None
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        
        for i in range(len(test_loader.dataset)):
            _data, _label, name = next(load_iter)
            if rgb_thermal_fusion:
                _data = [_data[0].cuda(), _data[1].cuda()]
            else:
                _data = _data.cuda()

            _label = _label.cuda()
            cls_label.append(int(_label))
            temp_predict = net(_data)['frame']

            cls_pre.append(1 if temp_predict.max() > 0.5 else 0)
            a_predict = temp_predict.mean(0).cpu().numpy()
            n = 16
            fpre_ = np.repeat(a_predict, n)
            if frame_predict is None:
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])
            temp_predict = torch.zeros((0)).cuda()
        
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)
        print(f"Step {step}: AUC={auc_score:.4f}, AP={ap_score:.4f}, Acc={accuracy:.4f}")

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)



