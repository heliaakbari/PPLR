import torch # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score
import torch.nn as nn


class Algorithm:
    def __init__(
        self,
        model,
        x_lb, dataset):
        self.gpu = 0
        self.ema = None
        # cv, nlp, speech builder different arguments
        self.model = model
        self.x_lb = x_lb
        self.dataset = dataset

    def evaluate(self, num_classes, batch_size, eval_dest='eval', out_key='logits', return_logits=False):
        """
        Evaluation function with Rank-k and mAP metrics.
        """
        self.model.eval()
        y_pred = []
        y_probs = []
        y_logits = []
        with torch.no_grad():
            numxxx = 0
            for i in range(0, self.x_lb.size(0), batch_size):
                x = self.x_lb[i:i+batch_size] 
                #print("x",x.shape)
                #print("y", y.shape)
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)

                if isinstance(self.model, nn.DataParallel):
                    logits = self.model.module.extract_global_classes(x) # Get model predictions
                else:
                    logits = self.model.extract_global_classes(x) # Get model predictions


                y_logits.append(logits.cpu().numpy())
                #print(logits.shape) #torch.Size([16, 2048])
                # Get probabilities
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                y_probs.extend(probs)
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            print("numxxx",numxxx)

        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        y_probs = np.array(y_probs)

        #self.ema.restore()
        self.model.train()

        print(f"y_pred:{y_pred}")
        num_unique = np.max(y_pred)
        print("Number of unique items:", num_unique)
        return y_pred, num_classes