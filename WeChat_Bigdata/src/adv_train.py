"""
对于每个x:
  1.计算x的前向loss、反向传播得到梯度
  2.根据embedding矩阵的梯度计算出r，并加到当前embedding上，相当于x+r
  3.计算x+r的前向loss，反向传播得到对抗的梯度，累加到第一步的梯度上
  4.将embedding恢复为第一步时的值
  5.根据第三步的梯度对参数进行更新
"""
import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'video' not in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'video' not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class PGD():
    """
    定义对抗训练方法PGD
    """
    def __init__(self, model, epsilon=1.0, alpha=0.3):
        # BERT模型
        self.model = model
        # 两个计算参数
        self.epsilon = epsilon
        self.alpha = alpha
        # 用于存储embedding参数
        self.emb_backup = {}
        # 用于存储梯度，与多步走相关
        self.grad_backup = {}

    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        """
        对抗
        :param emb_name: 模型中embedding的参数名
        :param is_first_attack: 是否是第一次攻击
        """
        # 循环遍历模型的每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 如果是第一次攻击
                if is_first_attack:
                    # 存储embedding参数
                    self.emb_backup[name] = param.data.clone()
                # 求梯度的范数
                norm = torch.norm(param.grad)
                # 如果范数不等于0
                if norm != 0:
                    # 计算扰动,param.grad / norm=单位向量相当于sgn符号函数
                    r_at = self.alpha * param.grad / norm
                    # 在原参数的基础上添加扰动
                    param.data.add_(r_at)
                    # 控制扰动后的模型参数值
                    # 投影到以原参数为原点，epsilon大小为半径的球上面
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self, emb_name='word_embeddings'):
        """
        将模型原本参数复原
        :param emb_name: 模型中embedding的参数名
        """
        # 循环遍历每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                # 取出模型真实参数
                param.data = self.emb_backup[name]
        # 清空emb_backup
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        """
        控制扰动后的模型参数值
        :param param_name:
        :param param_data:
        :param epsilon:
        """
        # 计算加了扰动后的参数值与原始参数的差值
        r = param_data - self.emb_backup[param_name]
        # 如果差值的范数大于epsilon
        if torch.norm(r) > epsilon:
            # 对差值进行截断
            r = epsilon * r / torch.norm(r)
        # 返回新的加了扰动后的参数值
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        """
        对梯度进行备份
        """
        # 循环遍历每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息
            if param.requires_grad:
                # 如果参数没有梯度
                if param.grad is None:
                    print("{} param has no grad !!!".format(name))
                    continue
                # 将参数梯度进行备份
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        """
        将梯度进行复原
        """
        # 循环遍历每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息
            if param.requires_grad:
                # 如果没有备份
                if name not in self.grad_backup:
                    continue
                # 如果备份了，就将原始模型参数梯度取出
                param.grad = self.grad_backup[name]

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)
# 需要使用对抗训练的时候，只需要添加五行代码
# 初始化
# if __name__ == '__main__':
#
#     fgm = FGM(model)
#     for batch_input, batch_label in data:
#         # 正常训练
#         loss = model(batch_input, batch_label)
#         loss.backward()  # 反向传播，得到正常的grad
#         # 对抗训练
#         fgm.attack()  # 在embedding上添加对抗扰动
#         loss_adv = model(batch_input, batch_label)
#         loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#         fgm.restore()  # 恢复embedding参数
#         # 梯度下降，更新参数
#         optimizer.step()
#         model.zero_grad()

# 实例初始化
# pgd = PGD(model)
# steps_for_at = 3
# for batch_input, batch_label in data:
#     # 正常训练
#     loss = model(batch_input, batch_label)
#     # 反向传播，得到正常的grad
#     loss.backward()
#     # 保存正常的梯度
#     pgd.backup_grad()
#     # PGD要走多步，迭代走多步
#     for t in range(steps_for_at):
#         # 在embedding上添加对抗扰动, first attack时备份param.data
#         pgd.attack(is_first_attack=(t == 0))
#         # 中间过程，梯度清零
#         if t != steps_for_at - 1:
#             optimizer.zero_grad()
#         # 最后一步，恢复正常的grad
#         else:
#             pgd.restore_grad()
#         # embedding参数被修改，此时，输入序列得到的embedding表征不一样
#     	loss_at = model(batch_input, batch_label)
#         # 对抗样本上的损失
#         loss_at = outputs_at[0]
#         # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#         loss_at.backward()
#     # 恢复embedding参数
#    	pgd.restore()
#     # 梯度下降，更新参数
#     optimizer.step()
#     # 将梯度清零
#     model.zero_grad()
