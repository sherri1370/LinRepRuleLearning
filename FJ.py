import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy
import random

def set_seed(seed=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

class ToTensor(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples


class StepActivation(nn.Module):
    def __init__(self, b):
        super(StepActivation).__init__()
        self.b = b
    def step(self, x):
        return torch.where(x >= self.b, torch.tensor(1.0), torch.tensor(0.0))

class SigmoidActivation(nn.Module):
    def __init__(self, a=1, b=0):
        super(SigmoidActivation, self).__init__()
        self.a = a
        self.b = b
    def sigmoid(self, x):
        return torch.sigmoid(self.a*x+self.b)
    
class RuleArchitecture(nn.Module):
    def __init__(self, input_size, output_size, conditions_size = [5, 5], regression_type = "logreg", q_act = SigmoidActivation()):
        super(RuleArchitecture, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conditions_size = conditions_size
        self.rules_list()
        self.regression_type = regression_type
        if self.regression_type != "logreg" and self.regression_type != "linreg":
            raise ValueError("Regression Type is not Defined: Use linreg or logreg.")
        self.q_act = q_act
        self.q_act_param = -q_act.b/q_act.a

    def rule(self, num_propos):
        self.propositions = nn.Linear(self.input_size, num_propos, bias=True)
        self.conjunctions = nn.Linear(num_propos, 1, bias=False)
        self.conjunctions.weight.data = torch.randint(2, size=self.conjunctions.weight.shape, dtype=torch.float32)
        # self.conjunctions.weight.data = torch.ones(size=self.conjunctions.weight.shape, dtype=torch.float32)*0.5
        self.rule_weights = nn.Linear(1, self.output_size, bias=True)
        return self

    def rules_list(self):
        self.rules = []
        for num_propos in self.conditions_size:
            self.rules.append(copy.deepcopy(self.rule(num_propos)))
        return self

    def forward(self, xin):
        xout = torch.tensor(0.0, dtype=torch.float32)
        for rule in self.rules:
            x = rule.propositions(xin)
            x = torch.sigmoid(x)
            x = x - 1  #balancing bias
            x = rule.conjunctions(x)
            x = self.q_act.sigmoid(x)
            xout = xout + rule.rule_weights(x)
        if self.regression_type == "logreg":
            xout = torch.sigmoid(xout)
        return xout
    
    def forward_step(self, xin):
        sa = StepActivation(b=self.q_act_param)
        xout = torch.tensor(0.0, dtype=torch.float32)
        for rule in self.rules:
            x = rule.propositions(xin)
            x = torch.sigmoid(x)
            x = x - 1  #balancing bias
            x = rule.conjunctions(x)
            x = sa.step(x)
            xout = xout + rule.rule_weights(x)
        if self.regression_type == "logreg":
            xout = torch.sigmoid(xout)
        return xout

    def forward_step_step(self, xin):
        sa_q = StepActivation(b=self.q_act_param)
        sa_p = StepActivation(b=0)
        xout = torch.tensor(0.0, dtype=torch.float32)
        for rule in self.rules:
            x = rule.propositions(xin)
            x = sa_p.step(x)
            x = x - 1  #balancing bias
            x = rule.conjunctions(x)
            x = sa_q.step(x)
            xout = xout + rule.rule_weights(x)
        if self.regression_type == "logreg":
            xout = torch.sigmoid(xout)
        return xout


    def predict(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            output = self.forward(input)
        return output.numpy()
    

    def predict_step(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            output = self.forward_step(input)
        return output.numpy()
    
    def predict_step_step(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            output = self.forward_step_step(input)
        return output.numpy()
    
    def binarise_model(self):
        for rule in self.rules:
            rule.conjunctions.weight.data = torch.where(rule.conjunctions.weight >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
        return self

    def prune_rules(self, percent=10):
        rule_bias = []
        rule_weights = []
        for rule in self.rules:
            propos_abs_weights = torch.cat((rule.propositions.weight, rule.propositions.bias.view(-1,1)), dim=1).abs().detach().cpu().numpy()
            rule_weights.append(rule.rule_weights.weight)
            rule_bias.append(rule.rule_weights.bias)
            threshold_propos = percent*propos_abs_weights.max(axis=1)/100
            prunned_propos_weight = torch.zeros(rule.propositions.weight.shape)
            prunned_propos_bias = torch.zeros(rule.propositions.bias.shape)
            for pi in range(rule.propositions.weight.shape[0]):
                prunned_propos_weight[pi,:] = torch.where(abs(rule.propositions.weight[pi,:])<= threshold_propos[pi] , torch.tensor(0.0), rule.propositions.weight[pi,:].data)
                prunned_propos_bias[pi] = torch.where(abs(rule.propositions.bias[pi])<= threshold_propos[pi] , torch.tensor(0.0), rule.propositions.bias[pi].data)
            rule.propositions.weight.data = prunned_propos_weight
            rule.propositions.bias.data = prunned_propos_bias

        rule_weights_vector = torch.tensor(rule_weights)
        threshold_rule_weights = percent*rule_weights_vector.max()/100
        for rule in self.rules:
            rule.rule_weights.weight.data = torch.where(abs(rule.rule_weights.weight) <= threshold_rule_weights , torch.tensor(0.0), rule.rule_weights.weight.data)

    
    def complexity(self):
        active_propos_nodes = 0
        active_conj_nodes = 0
        active_conj_nodes_check = 0
        non_zero_weights = torch.tensor(0)
        for rule in self.rules:
            if rule.rule_weights.weight != 0:
                propos_weights = rule.propositions.weight
                for pn_ind, pn in enumerate(rule.conjunctions.weight[0]):
                    if pn != 0:
                        if torch.count_nonzero(propos_weights[pn_ind, :]).item() > 0:
                            active_propos_nodes += 1
                            non_zero_weights += torch.count_nonzero(propos_weights[pn_ind,:])

                if active_propos_nodes > active_conj_nodes_check:
                    active_conj_nodes_check = active_propos_nodes
                    active_conj_nodes += 1
                
            c1 = active_conj_nodes
            c2 = active_propos_nodes + c1
            c3 = non_zero_weights.detach().numpy() + c2
        return c1, c2, c3

class LossFunction():
    def __init__(self, model, l1_propositions_penalty_lambda = 0, l1_rule_weights_penalty_lambda = 0, w_penalty_lambda = 0):
        self.model = model
        self.l1_propositions_penalty_lambda = l1_propositions_penalty_lambda
        self.l1_rule_weights_penalty_lambda = l1_rule_weights_penalty_lambda
        self.w_penalty_lambda = w_penalty_lambda

    def loss(self):
        if self.model.regression_type == "linreg":
            self.loss = nn.MSELoss()
        if self.model.regression_type == "logreg":
            self.loss = nn.BCELoss()
        return self.loss
    
    def l1_propositions_penalty(self):
        propositions_penalty = torch.tensor(0.0, dtype=torch.float32)
        if self.l1_propositions_penalty_lambda != 0:
            propositions_penalty = self.l1_propositions_penalty_lambda * (torch.sum(torch.abs(self.model.rules[-1].propositions.weight)) + torch.sum(torch.abs(self.model.rules[-1].propositions.bias)))
        return propositions_penalty
    
    def l1_rule_weights_penalty(self):
        rule_weights_penalty = torch.tensor(0.0, dtype=torch.float32)
        if self.l1_rule_weights_penalty_lambda != 0:
            rule_weights_penalty = self.l1_rule_weights_penalty_lambda * (sum([torch.sum(torch.abs(self.model.rules[i].rule_weights.weight)) for i in range(len(self.model.rules))]) + sum([torch.sum(torch.abs(self.model.rules[i].rule_weights.bias)) for i in range(len(self.model.rules))]))
        return rule_weights_penalty
    
    def w_penalty(self):
        conjunction_penalty = torch.tensor(0.0, dtype=torch.float32)
        if self.w_penalty_lambda != 0:
            conjunction_penalty = self.w_penalty_lambda * sum([torch.sum(torch.where(self.model.rules[i].conjunctions.weight < 0.5, torch.abs(self.model.rules[i].conjunctions.weight), torch.abs(self.model.rules[i].conjunctions.weight-1))) for i in range(len(self.model.rules))])
        return conjunction_penalty
        

class LearnRule():
    __constants__ = ['lr', 'dateloader_rate', 'gradient_threshold', 'n_epochs', 'batch_size']

    def __init__(self, learning_rate=0.0001, dataloader_rate=1, batch_size = 1, gradient_threshold=0.0001, n_epochs=20000, device='cpu'):
        super(LearnRule, self).__init__()
        self.lr = learning_rate
        self.dataloader_rate = dataloader_rate
        self.batch_size = batch_size
        self.gradient_threshold = gradient_threshold
        self.n_epochs = n_epochs
        self.device = device

    @staticmethod
    def get_dataloader(x, y, dataloader_rate=1):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dataset_ = TensorDataset(x,y)
        dl = DataLoader(dataset=dataset_, batch_size=int(len(x)*dataloader_rate), shuffle=True)
        return dl
    
    def get_conj_weights(self):
        with torch.no_grad():
            return self.conj_w_.numpy()
        
    def get_propos_weights(self):
        with torch.no_grad():
            return (self.propos_w_.numpy(), self.propos_b_.numpy())
        
    
    def fit(self, model, x, y, loss_func=None, refit = False):
        if refit is False:
            params = list(model.rules[-1].parameters())
            if len(model.rules)>1:
                for r in model.rules[:-1]:
                    params = params + list(r.parameters())
        if refit is True:
            params = list(model.rules[-1].rule_weights.parameters())
            if len(model.rules)>1:
                for r in model.rules[:-1]:
                    params = params + list(r.rule_weights.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(self.n_epochs/5), min_lr=1e-6)

        epoch_total_loss, epoch_train_loss, epoch_penalty_loss = [], [], []
    
        if loss_func == None:
            raise ValueError("Should define a loss criterion")
        lf = loss_func.loss()

        dl_ = self.get_dataloader(x, y, self.dataloader_rate)

        self.conj_w_ = torch.zeros(self.n_epochs,len(model.conjunctions.weight.flatten()))
        self.propos_w_ = torch.zeros(self.n_epochs,len(model.propositions.weight.flatten()))
        self.propos_b_ = torch.zeros(self.n_epochs,len(model.propositions.bias.flatten()))

        for epoch in range(self.n_epochs):
            self.conj_w_[epoch, :] = model.conjunctions.weight.flatten()
            self.propos_w_[epoch, :] = model.propositions.weight.flatten()
            self.propos_b_[epoch, :] = model.propositions.bias.flatten()

            trainloss, penaltyloss, totalloss, gradientnorm = [],[],[], []
            for batch_x_train, batch_y_train in dl_:
                optimizer.zero_grad()
                predictions = model.forward(batch_x_train)
                loss_value = lf(predictions, batch_y_train.view(-1,1))
                penalty_loss = loss_func.l1_propositions_penalty() + loss_func.w_penalty() + loss_func.l1_rule_weights_penalty()
                
                total_loss = loss_value + penalty_loss
                total_loss.backward()  
                optimizer.step()

                trainloss.append(loss_value.item())
                penaltyloss.append(penalty_loss.item())
                totalloss.append(total_loss.item())

                gradient_norms = []
                for param in model.rules[-1].parameters():
                    if param.grad is not None:
                        gradient_norms.append(torch.norm(param.grad).item())
                gradientnorm.append(gradient_norms)

            scheduler.step(np.mean(totalloss))
            epoch_total_loss.append(np.mean(totalloss))
            epoch_train_loss.append(np.mean(trainloss))
            epoch_penalty_loss.append(np.mean(penaltyloss))

            gn = np.array(gradientnorm).mean(axis=0)
            if all(norm < self.gradient_threshold for norm in gn):
                print(f'Early stopping at Epoch {epoch + 1} due to small gradients.')
                break
            if epoch+1 == self.n_epochs:
                print(f'epoch: {epoch+1}')
                

        return model, epoch_total_loss
    
    def refit(self, model, x, y, loss_func):
        return self.fit(model, x, y, loss_func, refit = True)

class Rules():
    def __init__(self, model, feature_name):
        self.feature_name = feature_name
        self.model = model
        self.get_rule_weights()

    def get_rule_weights(self):
        offset = 0
        for r in self.model.rules:
            offset += r.rule_weights.bias.detach().numpy()[0]
        self.rule_w = [offset]
        for r in self.model.rules:
            self.rule_w.append(r.rule_weights.weight.detach().numpy()[0][0])
        return self

    def rules_(self):
        rule_expression = [f"{self.rule_w[0]:.4f}"]
        for r_i, r in enumerate(self.model.rules):
            w = r.propositions.weight.detach().numpy().T
            t = r.propositions.bias.detach().numpy()
            conj_w = r.conjunctions.weight.detach().numpy()[0]
            propos_expression = []
            for p_i, p in enumerate(range(w.shape[1])):
                terms = []
                for i in range(len(self.feature_name)):
                    weight = w[i,p_i]
                    if weight != 0 and conj_w[p_i] != 0:
                        sign = '+' if weight > 0 and len(terms) > 0 else ''
                        terms.append(f"{sign}{weight:.4f}*{self.feature_name[i]}")
                if len(terms)>0 and p_i != w.shape[1]-1:
                    and_ = ' & '
                else:
                    and_ = ''
                if len(terms)>0:
                    propos_expression.append(f" ".join(terms) + f" >= {-t[0]:.4f}" + and_)
            if self.rule_w[r_i+1] != 0:
                rule_expression.append(f"{self.rule_w[r_i+1]:.4f} if " + " ".join(propos_expression))

        return rule_expression