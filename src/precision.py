import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.cnns import CNNModel
from src.dataset import CXRDataset
from src.runner import RunModelManager
from src.statistics import Statistics


class Precision:
    def __init__(self, config_parameters: dict):
        """
        Method is used as initializer of the class
        :param config_parameters: Configuration parameters of the project
        """
        self.configuration = config_parameters

    def get_runner_obj(self, model_name: str) -> RunModelManager:
        """
        Method is used to get main performer object according to the given model name
        :param model_name: string object to specify model name
        :return: RunModelManager object
        """
        return RunModelManager(self.configuration, model_name)

    def get_statistics_obj(self, model_name: str) -> tuple:
        """
        Method is used to collect labels dictionaries in a tuple
        :param model_name: string object to specify model name
        :return: tuple contains labels dictionary and idx to label object
        """
        stat_obj = Statistics(self.configuration, model_name)
        return stat_obj.configuration['labels'], stat_obj.configuration['id2label']

    def infer_image(self, model: CNNModel, image: torch.FloatTensor) -> int:
        """
        Model is used for inference process for provided image amd model
        :param model: specific model which will be used for inference
        :param image: image data as a Tensor
        :return:
        """
        image = torch.unsqueeze(image, dim=0)
        output = model(image.to(self.configuration['device']))
        output = output.view(-1, output.shape[-1])
        output = torch.argmax(output, dim=-1).item()
        return output

    def get_dataset(self, model_name: str) -> dict:
        """
        Method is used to collect dataset as a dictionary
        :param model_name: string object to specify model name
        :return: dictionary in which keys are image indexes and values are dict of image and its label
        """
        ds = CXRDataset(self.configuration, 'test', model_name, zero=True)
        required_set = {each['data']['idx']: {'image': each['data']['data'], 'label': each['label']} for each in ds}
        return required_set

    @staticmethod
    def get_combination(lab2id: dict) -> dict:
        """
        Method is used to create combination dictionary for given labels dict
        :param lab2id: dictionary in which keys are label and values are their indexes
        :return: dictionary which represent combination prediction and actual labels of data
        """
        comb_dict = dict()
        for label in lab2id.keys():
            for other in lab2id.keys():
                comb_dict[f'actual_{label}_pred_{other}'] = list()

        return comb_dict

    def get_model(self, model_name: str) -> CNNModel:
        """
        Method is used to get specific model according to the provided model name
        :param model_name: string object to specify model name
        :return: CNN model according to the specified model name
        """
        runner_obj = RunModelManager(self.configuration, model_name)
        runner_obj.load_model(is_best=True)
        runner_obj.model.eval()
        return runner_obj.model

    def infer_results(self) -> float:
        """
        Method is used to infer cascade model according to PH dataset
        :return: F1 score of the inference phase
        """
        ds = self.get_dataset('ph')
        ph_lab2id, ph_id2lab = self.get_statistics_obj('ph')
        vb_lab2id, vb_id2lab = self.get_statistics_obj('vb')
        om_lab2id, om_id2lab = self.get_statistics_obj('om')
        om_combination = self.get_combination(om_lab2id)
        ph_model = self.get_model('ph')
        vb_model = self.get_model('vb')
        for image_id, data in ds.items():
            prediction = self.infer_image(ph_model, data['image'])
            if ph_id2lab[prediction] == 'h_class':
                om_combination[f'actual_{image_id[0]}_class_pred_h_class'].append(image_id)
            else:
                vb_prediction = self.infer_image(vb_model, data['image'])
                vb_label = vb_id2lab[vb_prediction]
                om_combination[f'actual_{image_id[0]}_class_pred_{vb_label}'].append(image_id)

        om_stat = Statistics(self.configuration, 'om')

        result_dict = {'idx': list(), 'tgt': list(), 'pred': list()}
        for lab, idx in om_lab2id.items():
            for other, idx_other in om_lab2id.items():
                item = f'actual_{lab}_pred_{other}'
                result_dict['idx'].extend(om_combination[item])
                result_dict['tgt'].extend([idx] * len(om_combination[item]))
                result_dict['pred'].extend([idx_other] * len(om_combination[item]))

        cm = confusion_matrix(result_dict['tgt'], result_dict['pred'])
        f1 = f1_score(result_dict['tgt'], result_dict['pred'], average='macro')
        om_stat.plot_confusion_matrix(cm, cascade=True)
        return f1
