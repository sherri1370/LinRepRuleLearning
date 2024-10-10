import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lars
from scipy.special import expit
from sklearn.metrics import mean_squared_error, log_loss
import pandas as pd
from sklearn.model_selection import train_test_split



class IRLS_LARS:
    def __init__(self, t=0.01, max_num_nonzero_coef = 500, lambda_value=1, max_iter=100, tol=1e-6, fit_intercept=True): 
        self.max_iter = max_iter
        self.tol = tol
        self.t = t
        self.max_num_nonzero_coef = max_num_nonzero_coef
        self.lambda_value = lambda_value
        self.fit_intercept = fit_intercept

    def fit(self, x, y):
        n_samples, n_features = x.shape
        w_ = np.zeros(n_features)
        intercept_ = 0
        obj = log_loss(y, expit(x.dot(w_)+intercept_)) + self.lambda_value*(np.linalg.norm(w_) + abs(intercept_))

        for _ in range(self.max_iter):
            lambda_mat = np.diag([(expit(w_.dot(x[i,:])))*(1-expit(w_.dot(x[i,:]))) for i in range(n_samples)])
            z_vec = np.array([x[i,:].dot(w_)+(1-expit(y[i]*(w_.dot(x[i,:]))))*y[i]/(lambda_mat[i,i]+1e-10) for i in range(n_samples)])
            X_weighted = (lambda_mat**0.5).dot(x)
            y_weighted = (lambda_mat**0.5).dot(z_vec)
            lars = Lars(fit_intercept=self.fit_intercept, n_nonzero_coefs= self.max_num_nonzero_coef).fit(X_weighted, y_weighted)
            gamma_w = lars.coef_
            gamma_intercept = lars.intercept_

            best_obj_new = obj
            w_new_best = w_
            intercept_new_best = intercept_
            for t_candidate in np.concatenate(([0], np.logspace(-4, 0, 5))):  
                w_new = (1 - t_candidate) * w_ + t_candidate * gamma_w
                intercept_new = (1 - t_candidate) * intercept_ + t_candidate * gamma_intercept
                obj_new = log_loss(y, expit(x.dot(w_new) + intercept_new)) + self.lambda_value*(np.linalg.norm(w_new) + abs(intercept_new))
                if obj_new < best_obj_new:
                    best_t = t_candidate
                    best_obj_new = obj_new
                    w_new_best = w_new
                    intercept_new_best = intercept_new

            w_ = w_new_best
            intercept_ = intercept_new_best


            if abs(obj - best_obj_new) < self.tol:
                break

            obj = best_obj_new    

        self.coef_ = w_
        self.intercept_ = intercept_

        return self

    def predict(self, x):
        return expit(np.dot(x, self.coef_) + self.intercept_)

class LinRepRule_withLARS():
    def __init__(self, lambda_value=1, max_num_nonzero_coef = 500, epsilon_nzco=0.02, epsilon_prop=0.1, max_props=5):
        self.lambda_value = lambda_value
        self.epsilon_prop = epsilon_prop
        self.max_props = max_props
        self.rng = np.random.default_rng()
        self.max_num_nonzero_coef = max_num_nonzero_coef
        self.epsilon_nzco = epsilon_nzco

    def fit_signed(self, x, y):
        n, d = x.shape
        w = self.rng.normal(size=(d, self.max_props))
        t = self.rng.normal(size=(self.max_props))
        self.num_props_ = 0
        for p in range(self.max_props):
            print(f'fitting prop {p+1}')
            selected = self._predict(x, w[:,:self.num_props_], t[:self.num_props_]).astype(bool)
            old_obj = sum(y[selected])/np.sqrt((sum(selected)))
            if len(y[selected])==sum(y[selected]>=0) or len(y[selected])==sum(y[selected]<0):
                print('all selected samples are positive or negative')
                break

            best_new_obj = 1e-10
            w_temp = w
            t_temp = t
            self.num_props_ += 1
            best_nz_coef = 1
            for i in np.arange(1, self.max_num_nonzero_coef+1):
                lr = IRLS_LARS(lambda_value=self.lambda_value, max_num_nonzero_coef=i, fit_intercept=True).fit(x[selected],y[selected]>=0)
                
                w_temp[:, self.num_props_-1] = lr.coef_
                t_temp[self.num_props_-1] = lr.intercept_
                new_selected = self._predict(x, w_temp[:,:self.num_props_], t_temp[:self.num_props_]).astype(bool)
                new_obj_temp = sum(y[new_selected])/np.sqrt((sum(new_selected)))
                # print(f'best temp : {new_obj_temp} and best : {best_new_obj} and ratio : {(new_obj_temp-best_new_obj)/best_new_obj}')
                if (new_obj_temp-best_new_obj)/best_new_obj >= self.epsilon_nzco:
                    
                    best_new_obj = new_obj_temp
                    w[:, self.num_props_-1] = lr.coef_
                    t[self.num_props_-1] = lr.intercept_
                    best_nz_coef = i
                
            print(f'best num nonzero coef is {best_nz_coef}')
            if abs((best_new_obj - old_obj)/old_obj) <= self.epsilon_prop:
                self.num_props_ -= 1
                print(f'rejected')
                best_new_obj = old_obj
                break

        return w[:,:self.num_props_], t[:self.num_props_], best_new_obj

    def fit(self, x, y):
        w_pos, t_pos, obj_pos = self.fit_signed(x, y)
        w_neg, t_neg, obj_neg = self.fit_signed(x, -y)
        
        self.w_ = w_pos
        self.t_ = t_pos
        if obj_neg > obj_pos:
            print('negative')
            self.w_ = -w_neg
            self.t_ = -t_neg
        else:
            print('positive')

        return self
    
    @staticmethod
    def _predict(x,w,t):
        l = x@w
        s = l>=-t
        return np.prod(s, axis=1)

    def predict(self, x):
        return self._predict(x, self.w_, self.t_)

class LinRepRuleEnsemble(BaseEstimator):

    def __init__(self, lambda_value_rule_weights = 0, lambda_value_prop=1, epsilon_prop=0.1, max_num_nonzero_coef = 500, epsilon_nzco = 0.02, max_props=5, num_rules=3, optim_problem = 'linreg'):
        self.lambda_value_prop = lambda_value_prop
        self.epsilon_prop = epsilon_prop
        self.epsilon_nzco = epsilon_nzco
        self.max_props = max_props
        self.num_rules = num_rules
        self.max_num_nonzero_coef = max_num_nonzero_coef
        self.optim_problem = optim_problem
        self.lambda_value_rule_weights= lambda_value_rule_weights


    def fit(self, x,y):
        self.feature_name = x.columns
        x = x.to_numpy()
        y = y.to_numpy()
        self.conditions_ = []
        self.weights_ = np.array([y.mean()])
        for _ in range(self.num_rules):
            if self.optim_problem == 'linreg':
                g = y - self.predict(x)
            else:
                g = (1-y)/(1-self.predict(x)) - y/self.predict(x)
            self.conditions_.append(LinRepRule_withLARS(lambda_value=self.lambda_value_prop, max_num_nonzero_coef = self.max_num_nonzero_coef, epsilon_nzco=self.epsilon_nzco, epsilon_prop= self.epsilon_prop, max_props=self.max_props).fit(x,g))
            c = self.compute_C(x)
            g = c.T.dot(c)
            rank = np.linalg.matrix_rank(g)
            if rank < min(g.shape):
                self.conditions_.pop()
                break
            if self.optim_problem == 'linreg':
                g = c.T.dot(c) + self.lambda_value_rule_weights*np.eye(c.shape[1])
                self.weights_ = np.linalg.solve(g,c.T.dot(y))
            elif self.optim_problem == 'logreg':
                logr = LogisticRegression(fit_intercept=False, solver='newton-cg').fit(c,y)
                self.weights_ = logr.coef_[0]

        return self
    
    def compute_C(self, x):
        res = np.zeros(shape=(len(x), len(self.conditions_)+1))
        res[:,0] = 1
        for i, c in enumerate(self.conditions_):
            res[:, i+1] = c.predict(x)
        return res

    
    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        pred = self.compute_C(x) 
        if self.optim_problem == 'linreg':
            out = pred.dot(self.weights_)
        else:
            out = expit(pred.dot(self.weights_))
        return out
    
    def complexity_c1(self):
        return len(self.conditions_)
    
    def complexity_c2(self):
        return self.complexity_c1() + sum([c.w_.shape[1] for c in self.conditions_])
    
    def complexity_c3(self):
        return self.complexity_c2() + sum([np.count_nonzero(c.w_) for c in self.conditions_])
    
    def rules_(self):
        rule_w = self.weights_
        rule_expression = [f"{rule_w[0]:.4f}"]
        for r_i, r in enumerate(self.conditions_):
            w = r.w_
            t = r.t_
            propos_expression = []
            for p_i, p in enumerate(range(w.shape[1])):
                terms = []
                for i in range(len(self.feature_name)):
                    weight = w[i,p_i]
                    if weight != 0:
                        sign = '+' if weight > 0 and len(terms) > 0 else ''
                        terms.append(f"{sign}{weight:.4f}*{self.feature_name[i]}")
                if w.shape[1] > 1 and p_i != w.shape[1]-1:
                    and_ = ' & '
                else:
                    and_ = ''
                propos_expression.append(f" ".join(terms) + f" >= {-t[0]:.4f}" + and_)

            rule_expression.append(f"{rule_w[r_i+1]:.4f} if " + " ".join(propos_expression))

        return rule_expression
