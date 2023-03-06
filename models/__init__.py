import torch
import torch.nn as nn

from .backbones import AVA_backbone
from .necks import AVA_neck
from .heads import AVA_head


class AVA_model(nn.Module):
    def __init__(self, config):
        super(AVA_model, self).__init__()
        self.config = config

        self.backbone = AVA_backbone(config.backbone)
        self.neck = AVA_neck(config.neck)
        self.head = AVA_head(config.head)

    def forward(self, data, evaluate=False):
        if not evaluate:  # train mode
            i_b = {'clips': data['clips']}
            o_b = self.backbone(i_b)

            i_n = {'aug_info': data['aug_info'], 'labels': data['labels'],
                   'filenames': data['filenames'], 'mid_times': data['mid_times']}
            o_n = self.neck(i_n, evaluate)

            if o_n['num_rois_ps'] == 0:
                return {'outputs': None, 'targets': o_n['targets'],
                        'num_rois': 0, 'filenames': o_n['filenames'],
                        'mid_times': o_n['mid_times'], 'bboxes': o_n['bboxes'],
                        'bg_feats': None,
                        'ps_feats': None, 'num_rois_ps': 0,
                        'obj_feats': None,
                        'num_rois_obj': 0, 'reduce_bg_feats': None
                        }

            i_h = {'features': o_b['features'],
                   'rois': o_n['rois'],
                   'num_rois': o_n['num_rois'],
                   'roi_ids': o_n['roi_ids'],

                   'sizes_before_padding': o_n['sizes_before_padding'], 'person_ids': o_n['person_ids'],

                   'rois_ps': o_n['rois_ps'],
                   'num_rois_ps': o_n['num_rois_ps'],
                   'roi_ps_ids': o_n['roi_ps_ids'],

                   'rois_obj': o_n['rois_obj'],
                   'num_rois_obj': o_n['num_rois_obj'],
                   'roi_obj_ids': o_n['roi_obj_ids']
                   }

            o_h = self.head(i_h)

            return {'outputs': o_h['outputs'], 'targets': o_n['targets'],
                    'num_rois': o_n['num_rois'], 'filenames': o_n['filenames'],
                    'mid_times': o_n['mid_times'], 'bboxes': o_n['bboxes'], 'bg_feats': o_h['bg_feats'],
                    'ps_feats': o_h['ps_feats'], 'num_rois_ps': o_n['num_rois_ps'], 'obj_feats': o_h['obj_feats'],
                    'num_rois_obj': o_n['num_rois_obj'], 'reduce_bg_feats': o_h['reduce_bg_feats'],
                    'B_alpha': o_h['B_alpha'], 'B_beta': o_h['B_beta']}

        # ==============================================================================================================
        # eval mode
        assert not self.training
        noaug_info = [{'crop_box': [0., 0., 1., 1.], 'flip': False, 'pad_ratio': [1., 1.]}] * len(data['labels'])

        i_n = {'aug_info': noaug_info, 'labels': data['labels'],
               'filenames': data['filenames'], 'mid_times': data['mid_times']}
        o = self.neck(i_n, evaluate)

        output_list = [None] * len(o['filenames'])
        B_alpha_list = [None] * len(o['filenames'])
        B_beta_list = [None] * len(o['filenames'])
        cnt_list = [0] * len(o['filenames'])

        for no in range(len(data['clips'])):
            i_b = {'clips': data['clips'][no]}
            o_b = self.backbone(i_b)

            i_n = {'aug_info': data['aug_info'][no], 'labels': data['labels'],
                   'filenames': data['filenames'], 'mid_times': data['mid_times']}
            o_n = self.neck(i_n, evaluate)

            if o_n['num_rois'] == 0:
                continue
            ids = o_n['bbox_ids']

            i_h = {'features': o_b['features'],
                   'rois': o_n['rois'],
                   'num_rois': o_n['num_rois'],
                   'roi_ids': o_n['roi_ids'],

                   'sizes_before_padding': o_n['sizes_before_padding'], 'person_ids': o_n['person_ids'],

                   'rois_ps': o_n['rois_ps'],
                   'num_rois_ps': o_n['num_rois_ps'],
                   'roi_ps_ids': o_n['roi_ps_ids'],

                   'rois_obj': o_n['rois_obj'],
                   'num_rois_obj': o_n['num_rois_obj'],
                   'roi_obj_ids': o_n['roi_obj_ids']
                   }
            o_h = self.head(i_h)

            outputs = o_h['outputs']
            B_alpha = o_h['B_alpha']
            B_beta = o_h['B_beta']

            for idx in range(o_n['num_rois']):
                if cnt_list[ids[idx]] == 0:
                    output_list[ids[idx]] = outputs[idx]
                    B_alpha_list[ids[idx]] = B_alpha[idx]
                    B_beta_list[ids[idx]] = B_beta[idx]
                else:
                    output_list[ids[idx]] += outputs[idx]
                    B_alpha_list[ids[idx]] += B_alpha[idx]
                    B_beta_list[ids[idx]] += B_beta[idx]
                cnt_list[ids[idx]] += 1

        num_rois, filenames, mid_times, bboxes, targets, outputs = 0, [], [], [], [], []
        B_alpha, B_beta, gt_uncertainties = [], [], []

        for idx in range(len(o['filenames'])):
            if cnt_list[idx] == 0:
                continue
            num_rois += 1
            filenames.append(o['filenames'][idx])
            mid_times.append(o['mid_times'][idx])
            bboxes.append(o['bboxes'][idx])
            targets.append(o['targets'][idx])
            if o['gt_uncertainties'] is not None:
                gt_uncertainties.append(o['gt_uncertainties'][idx])
            else:
                gt_uncertainties.append(None)
            outputs.append(output_list[idx] / float(cnt_list[idx]))
            B_alpha.append(B_alpha_list[idx] / float(cnt_list[idx]))
            B_beta.append(B_beta_list[idx] / float(cnt_list[idx]))

        if num_rois == 0:
            return {'outputs': None, 'targets': None, 'num_rois': 0,
                    'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'B_alpha': None, 'B_beta': B_beta,
                    'gt_uncertainties': None}

        final_outputs = torch.stack(outputs, dim=0)
        final_B_alpha = torch.stack(B_alpha, dim=0)
        final_B_beta = torch.stack(B_beta, dim=0)
        final_targets = torch.stack(targets, dim=0)
        try:
            final_gt_uncertainties = torch.tensor(gt_uncertainties)
        except RuntimeError:
            print("Warning: final_gt_uncertainties got None type.")
            final_gt_uncertainties = None


        return {'outputs': final_outputs, 'targets': final_targets, 'num_rois': num_rois,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'B_alpha': final_B_alpha,
                'B_beta': final_B_beta, 'gt_uncertainties': final_gt_uncertainties}

    def train(self, mode=True):
        super(AVA_model, self).train(mode)

        if mode and self.config.get('freeze_bn', False):
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.backbone.apply(set_bn_eval)
