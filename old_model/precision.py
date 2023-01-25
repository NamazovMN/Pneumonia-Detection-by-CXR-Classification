import matplotlib.pyplot as plt
import torch
import numpy as np
from utilities import *
from DatasetGen import Xrays
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import pandas as pd
class Precision:
    def __init__(self, save_path, om_model, vb_model, np_model, om_test, vb_test, np_test, experiment_path, experiment_num, device):
        self.save_path = save_path
        self.om_model = om_model
        self.vb_model = vb_model
        self.np_model = np_model
        self.om_test = om_test
        self.vb_test = vb_test
        self.np_test = np_test
        self.experiment_path = experiment_path
        self.experiment_num = experiment_num
        self.device = device

    def predict(self, model, test):
        correct = 0
        prediction_data = list()
        labels_list = list()
        with torch.no_grad():
            model.eval()
            for each_data in test:
                data = each_data['data'].to(self.device)
                labels = each_data['labels'].to(self.device)
                labels_list += [each.item() for each in labels]
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)

                prediction_data += [each.item() for each in predicted]
                correct += (predicted == labels).sum().item()

        return labels_list, prediction_data

    def generate(self):
        np_lab, np_pred = self.predict(self.np_model, self.np_test)
        len_n = 0
        for each in np_lab:
            if each == 0:
                len_n += 1


        count_nn = 0
        count_np = 0
        count_pp = 0
        count_pn = 0
        np_idxs = list()
        pp_idxs = list()
        pn_idxs = list()
        for idx in range(len(np_lab)):
            if np_lab[idx] == np_pred[idx]:
                if np_lab[idx] == 0:
                    count_nn += 1
                else:
                    count_pp += 1
                    pp_idxs.append(idx)
            else:
                if np_lab[idx] == 0:
                    count_np += 1
                    np_idxs.append(idx)
                else:
                    count_pn += 1
                    pn_idxs.append(idx)

        pp_res_dict = self.process_pp(pp_idxs, len_n)
        np_res_dict = self.process_np(np_idxs)
        pn_res_dict = self.process_pn(pn_idxs, len_n)
        om_labs, om_preds = self.process_om()

        conf_mat_om = confusion_matrix(om_labs, om_preds)

        conf_matrix = np.array([
            [count_nn, np_res_dict["virus"], np_res_dict["bacteria"]],
            [pn_res_dict["virus"], pp_res_dict["count_vv"], pp_res_dict["count_vb"]],
            [pn_res_dict["bacteria"], pp_res_dict["count_bv"], pp_res_dict["count_bb"]]]
        )

        conf_list_labels = ['Healthy', 'Viral Pn.', 'Bacterial Pn.']
        df_cm_ccnn = pd.DataFrame(conf_matrix, index=[i for i in conf_list_labels],
                             columns=[i for i in conf_list_labels])
        # plt.figure(figsize=(3, 3))
        sn.heatmap(df_cm_ccnn, annot=True, fmt='d', cmap='PuBu')
        plt.title('Cascade Model')
        plt.show()

        df_cm_dcnn = pd.DataFrame(conf_mat_om, index=[i for i in conf_list_labels],
                             columns=[i for i in conf_list_labels])
        sn.heatmap(df_cm_dcnn, annot=True, fmt='d', cmap='PuBu')
        plt.title('Direct Model')
        plt.show()
        return conf_matrix, conf_mat_om, om_labs, om_preds


    def process_pp(self, pp_idxs, max_val):
        cor_vb_idxs = [each - max_val for each in pp_idxs]
        vb_test_data = np.load(os.path.join(self.save_path, 'vb_test_data.npy'))
        vb_test_labels = np.load(os.path.join(self.save_path, 'vb_test_labels.npy'))
        req_labels = np.array([vb_test_labels[each_idx] for each_idx in cor_vb_idxs])
        req_data = np.array([vb_test_data[each_idx] for each_idx in cor_vb_idxs])
        req_dataset = Xrays({'data': req_data, 'labels': req_labels})
        req_loader = DataLoader(req_dataset, shuffle=False, batch_size=32)
        vb_pp_lab, vb_pp_pred = self.predict(self.vb_model, req_loader)

        count_vv = 0
        count_bb = 0
        count_vb = 0
        count_bv = 0
        for idx in range(len(vb_pp_lab)):
            if vb_pp_lab[idx] == vb_pp_pred[idx]:
                if vb_pp_lab[idx] == 0:
                    count_vv += 1
                else:
                    count_bb += 1
            else:
                if vb_pp_lab[idx] == 0:
                    count_vb += 1
                else:
                    count_bv += 1

        result = {'count_vv': count_vv, 'count_vb': count_vb, 'count_bv': count_bv, 'count_bb': count_bb}

        return result

    def process_np(self, np_idxs):
        np_test_data = np.load(os.path.join(self.save_path, 'np_test_data.npy'))
        np_test_labels = np.load(os.path.join(self.save_path, 'np_test_labels.npy'))

        req_data = np.array([np_test_data[idx] for idx in np_idxs])
        req_labels = np.array([np_test_labels[idx] for idx in np_idxs])

        req_dataset = Xrays({'data': req_data, 'labels': req_labels})
        req_loader = DataLoader(req_dataset, shuffle=False, batch_size=32)

        _, vb_np_pred = self.predict(self.vb_model, req_loader)
        count_nv = 0
        count_nb = 0

        for each in vb_np_pred:
            if each == 0:
                count_nv += 1
            else:
                count_nb += 1

        result = {'virus': count_nv, 'bacteria': count_nb}

        return result

    def process_pn(self, pn_idxs, max_val):
        vb_cor_idxs = [each_idx - max_val for each_idx in pn_idxs]

        vb_test_labels = np.load(os.path.join(self.save_path, 'vb_test_labels.npy'))

        original_labels = [vb_test_labels[idx] for idx in vb_cor_idxs]

        count_vn = 0
        count_bn = 0

        for each in original_labels:
            if each == 0:
                count_vn += 1
            else:
                count_bn += 1

        result = {'virus': count_vn, 'bacteria': count_bn}

        return result


    def process_om(self):
        vb_test_data = np.load(os.path.join(self.save_path, 'vb_test_data.npy'))
        vb_test_labels = np.load(os.path.join(self.save_path, 'vb_test_labels.npy'))
        np_test_data = np.load(os.path.join(self.save_path, 'np_test_data.npy'))
        np_test_labels = np.load(os.path.join(self.save_path, 'np_test_labels.npy'))

        om_req_data = list()
        om_req_labels = list()
        for each_idx in range(len(np_test_labels)):
            if np_test_labels[each_idx] == 0:
                om_req_data.append(np_test_data[each_idx])
                om_req_labels.append(0)

        for each_idx in range(len(vb_test_labels)):
            om_req_data.append(vb_test_data[each_idx])
            if vb_test_labels[each_idx] == 0:
                om_req_labels.append(1)
            else:
                om_req_labels.append(2)

        om_req_dataset = Xrays({'data': om_req_data, 'labels': om_req_labels})

        om_req_loader = DataLoader(om_req_dataset, shuffle=False)

        om_labs, om_preds = self.predict(self.om_model, om_req_loader)

        return om_labs, om_preds

    def compute_precision_cascade(self):
        confusion_mat_cascade, confusion_mat_om, om_labs, om_preds = self.generate()

        TP_0 = confusion_mat_cascade[0, 0]
        FP_0 = np.sum(confusion_mat_cascade[0, 1:])
        TN_0 = np.sum(confusion_mat_cascade[1, 1:]) + np.sum(confusion_mat_cascade[2, 1:])
        FN_0 = np.sum(confusion_mat_cascade[1:, 0])

        precision_0 = TP_0 / (TP_0 + FP_0)
        recall_0 = TP_0 / (TP_0 + FN_0)
        f1_score0 = 2 * ((precision_0 * recall_0) / (precision_0 + recall_0))

        TP_1 = confusion_mat_cascade[1, 1]
        FP_1 = confusion_mat_cascade[1, 0] + confusion_mat_cascade[1, 2]
        TN_1 = confusion_mat_cascade[0, 0] + confusion_mat_cascade[0, 2] + confusion_mat_cascade[2, 0] + confusion_mat_cascade[2,2]
        FN_1 = confusion_mat_cascade[0, 1] + confusion_mat_cascade[2, 1]

        precision_1 = TP_1 / (TP_1 + FP_1)
        recall_1 = TP_1 / (TP_1 + FN_1)
        f1_score1 = 2 * ((precision_1 * recall_1) / (precision_1 + recall_1))

        TP_2 = confusion_mat_cascade[2, 2]
        FP_2 = confusion_mat_cascade[2, 0] + confusion_mat_cascade[2, 1]
        TN_2 = confusion_mat_cascade[0, 0] + confusion_mat_cascade[0, 1] + confusion_mat_cascade[1, 0] + confusion_mat_cascade[1, 1]
        FN_2 = confusion_mat_cascade[0, 2] + confusion_mat_cascade[1, 2]

        precision_2 = TP_2 / (TP_2 + FP_2)
        recall_2 = TP_2 / (TP_2 + FN_2)
        f1_score2 = 2 * ((precision_2 * recall_2) / (precision_2 + recall_2))



        total_TP = TP_0 + TP_1 + TP_2
        total_FP = FP_0 + FP_1 + FP_2
        total_TN = TN_0 + TN_1 + TN_2
        total_FN = FN_0 + FN_1 + FN_2

        precision_total = total_TP / (total_TP + total_FN)
        recall_total = total_TP / (total_TP + total_FP)
        micro_f1_score = 2 * ((precision_total * recall_total)/(precision_total + recall_total))


        macro_f1_score = (f1_score0 + f1_score1 + f1_score2)/3
        macro_f1_om = f1_score(om_labs, om_preds, average='macro')
        micro_f1_om = f1_score(om_labs, om_preds, average='micro')

        print(f'Cascade Model Results: Micro F1: {micro_f1_score:.4f}; Macro F1: {macro_f1_score:.4f}')
        print(f'Direct Model Results: Micro F1: {micro_f1_om:.4f}; Macro F1: {macro_f1_om:.4f}')
        f1_results = {
            'microF1_om': micro_f1_om,
            'macroF1_om': macro_f1_om,
            'microF1_cascade': micro_f1_score,
            'macroF1_cascade': macro_f1_score
        }
        file_name = os.path.join(self.experiment_path, f'f1_results_{self.experiment_num}.json')
        with open(file_name, 'w') as results:
            json.dump(f1_results, results)

        conf_file_cascade = os.path.join(self.experiment_path, f'confusion_cascade_{self.experiment_num}.npy')
        conf_file_direct = os.path.join(self.experiment_path, f'confusion_direct_{self.experiment_num}.npy')



        np.save(conf_file_cascade, confusion_mat_cascade)
        np.save(conf_file_direct, np.array(confusion_mat_om))
