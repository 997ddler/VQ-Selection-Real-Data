import torch



mean_1 = torch.tensor(0.0)
std_1 = torch.tensor(1.0)

mean_2 = torch.tensor(0.5)
std_2 = torch.tensor(1.0)

normal_dist_1 = torch.distributions.Normal(mean_1, std_1)
normal_dist_2 = torch.distributions.Normal(mean_2, std_2)

