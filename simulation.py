
import torch
import math
from tqdm import tqdm
import json


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SCALER = 16

log_n = 14
log_m = 14
n = pow(2, log_n) #Number of buyers
m = pow(2, log_m) #Number of goods
min_nm = min(n,m)
T = 16 #Iterations
rounds = 15 #Replication

result = {}


def pr(B, v, T, n, m):
    b = torch.ones((n,m), device=DEVICE)/m * B.unsqueeze(1)
    omega = []
    for _ in tqdm(range(T)):
        p = torch.sum(b, 0)
        w = v * b / p
        u = torch.sum(w, 1)
        omega.append(torch.sum(B * torch.log(u)).item())
        b = w / u.unsqueeze(1) * B.unsqueeze(1)
    del b
    return omega


def get_opt(B, v, n, m):
    b = torch.ones((n,m), device=DEVICE)/m * B.unsqueeze(1)
    for _ in tqdm(range(1001)):
        p = torch.sum(b, 0)
        w = v * b / p
        u = torch.sum(w, 1)
        b = w / u.unsqueeze(1) * B.unsqueeze(1)
    del b
    return torch.sum(B * torch.log(u)).item()


def project_onto_simplex(b, B, m):
    sb = b.sort(descending=True)[0]
    cb = sb.cumsum(dim = -1)
    cb = (cb - B.unsqueeze(1))/torch.arange(1, m+1, device=DEVICE)
    K = torch.sum((sb - cb) > 0, dim = -1).long() - 1
    del sb
    mu = cb.take_along_dim(K.unsqueeze(1), dim=-1)
    del cb
    del K
    b = b - mu
    del mu
    return b.clamp(min=0)

def grad_f_linear_shmyrev(b, v): 
    gg = 1 - torch.log(v/b.sum(dim = 0)).nan_to_num(neginf=1)
    return gg

def pgd(B, v, T, n, m):
    b = torch.ones((n,m), device=DEVICE)/m * B.unsqueeze(1)
    denom_vec = v.sum(dim = 1)
    temp_mat = (v.T * B / denom_vec).T
    del denom_vec
    p_lower_linear = temp_mat.max(dim = 0)[0]
    del temp_mat
    Lf_linear_shmyrev = n/p_lower_linear.min()/1000
    del p_lower_linear

    omega = []
    for t in tqdm(range(T+1)):
        p = torch.sum(b, 0)
        w = v * b / p
        del p
        u = torch.sum(w, 1)
        del w
        omega.append(torch.sum(B * torch.log(u)).item())
        del u
        if t == 0:
            b = b*(v>0)
        b = b - grad_f_linear_shmyrev(b, v) / Lf_linear_shmyrev
        b = project_onto_simplex(b, B, m)
    del b
    return omega


def amp_est(p, M, num_bins, samples_per_bin):
    arr = p.sqrt().arcsin()/torch.pi
    arr = torch.arange(float(M), device=DEVICE).div(M).unsqueeze(1).expand((-1, p.shape[0])) - arr
    arr = torch.abs(arr)
    arr = torch.minimum(arr, 1 - arr)
    num = torch.sin(M * torch.pi * arr).square()
    den = M * M * torch.sin(torch.pi * arr).square()
    arr = torch.transpose(torch.nan_to_num(num/den, 1), 0, 1)
    arr = torch.multinomial(arr, num_bins * samples_per_bin, replacement=True)
    arr = torch.sin(arr / M * torch.pi).square()
    return arr.view(-1, num_bins, samples_per_bin).mean(dim = -1).median(dim=-1)[0]


def qfpr(B, v, T, n, m, min_nm):
    b = torch.ones((n,m), device=DEVICE)/m * B.unsqueeze(1)
    iters = round(pow(min_nm * T, 1/2.0))
    mul = pow(2, math.ceil(math.log2(iters)))
    omega_est = []

    for t in tqdm(range(iters + 1)):
        p = torch.sum(b, 0)
        w = v * b / p
        u = torch.sum(w, 1)
        omega_est.append(torch.sum(B * torch.log(u)).item())
        if t == iters:
            break
        del w
        del u

        b_max = b.max(0)[0]
        p = amp_est(p/n/b_max, mul * int(math.sqrt(n))/SCALER, 3, 7)*n*b_max
        del b_max
        w = v * b / p
        del p
        w_max = w.max(1)[0]
        u = torch.sum(w, 1)
        u = amp_est(u/m/w_max, mul * int(math.sqrt(m))/SCALER, 3, 7)*m*w_max
        del w_max
        b = w / u.unsqueeze(1) * B.unsqueeze(1)
        del w
        del u
        
    del b
    return omega_est


B = torch.rand(n, device=DEVICE)
B = B/torch.sum(B)
v = torch.rand(n, m, device=DEVICE)
pr_omega = pr(B, v, T, n, m)
pgd_omega = pgd(B, v, T, n, m)
opt = get_opt(B, v, n, m)
qfpr_omega = []
for i in range(rounds):
    qfpr_omega.append(qfpr(B, v, T, n, m, min_nm))
entry = {"pr": pr_omega, 
         "pgd": pgd_omega,
         "qfpr": qfpr_omega,
         "stop": opt}
result["uniform"] = entry


B = torch.ones(n, device=DEVICE)/n
v = torch.rand(n, m, device=DEVICE)
pr_omega = pr(B, v, T, n, m)
pgd_omega = pgd(B, v, T, n, m)
opt = get_opt(B, v, n, m)
qfpr_omega = []
for i in range(rounds):
    qfpr_omega.append(qfpr(B, v, T, n, m, min_nm))
entry = {"pr": pr_omega, 
         "pgd": pgd_omega,
         "qfpr": qfpr_omega,
         "stop": opt}
result["uniform_ceei"] = entry


B = torch.nn.init.trunc_normal_(torch.empty(n, device=DEVICE),0.5, 0.25, 0, 1)
B = B/torch.sum(B) #Random allocation of budget
v = torch.nn.init.trunc_normal_(torch.empty(n, m, device=DEVICE), 0.5, 0.25, 0, 1)
pr_omega = pr(B, v, T, n, m)
pgd_omega = pgd(B, v, T, n, m)
opt = get_opt(B, v, n, m)
qfpr_omega = []
for i in range(rounds):
    qfpr_omega.append(qfpr(B, v, T, n, m, min_nm))
entry = {"pr": pr_omega, 
         "pgd": pgd_omega,
         "qfpr": qfpr_omega,
         "stop": opt}
result["normal"] = entry


B = torch.ones(n, device=DEVICE) / n
v = torch.nn.init.trunc_normal_(torch.empty(n, m, device=DEVICE), 0.5, 0.25, 0, 1)
pr_omega = pr(B, v, T, n, m)
pgd_omega = pgd(B, v, T, n, m)
opt = get_opt(B, v, n, m)
qfpr_omega = []
for i in range(rounds):
    qfpr_omega.append(qfpr(B, v, T, n, m, min_nm))
entry = {"pr": pr_omega, 
         "pgd": pgd_omega,
         "qfpr": qfpr_omega,
         "stop": opt}
result["normal_ceei"] = entry


with open('data.json', 'w') as f:
    json.dump(result, f)
