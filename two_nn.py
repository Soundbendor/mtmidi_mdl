import torch

def get_dist_idx(_i,_j, N):
    i = min(_i,_j)
    j = max(_i,_j)
    return N * i + j - ((i+2) * (i+1)) // 2

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
def get_dist_idxs(cur_idxs, idxs, N, device='cpu'):
    ipts = None
    num_idxs = cur_idxs.shape[0]
    """
    for cur_idx in cur_idxs:
        _ipts = torch.vstack((torch.ones(N-1, dtype=torch.int64, device=device) * cur_idx, torch.hstack((idxs[:cur_idx], idxs[(cur_idx+1):])))).sort(axis=0,descending=False).values
        if ipts == None:
            ipts = _ipts
        else:
            ipts = torch.hstack((ipts, _ipts))
    #ipts = torch.vstack((torch.tensor([cur_idx], dtype=torch.int64, device=device).repeat(N-1), torch.hstack((idxs[:cur_idx], idxs[(cur_idx+1):])))).sort(axis=0,descending=False).values
    """
    _ipts = torch.vstack((cur_idxs.repeat(N,1).T.flatten(), idxs.repeat(num_idxs)))
    ipts = _ipts.T[_ipts[0] != _ipts[1]].sort(axis=1).values.T
    ret = N * ipts[0] + ipts[1] - ((ipts[0]+2) * (ipts[0]+1)) // 2
    return ret.reshape(num_idxs,N-1)


def get_mus(idxs, N, dist_vec, batch_size = 64, device='cpu'):
    #log_mus = torch.log(mus.index_select(0, idxs[idxs != max_idx])).unsqueeze(1)
    last_slice = 0
    shortest = None
    while last_slice < N:
        cur_slice = last_slice + batch_size
        dist_idxs = get_dist_idxs(idxs[last_slice:cur_slice], idxs, N)
        cur_dists = dist_vec[dist_idxs]
        _shortest = cur_dists.sort(axis=1, descending=False).values[:,:2]
        if shortest == None:
            shortest = _shortest
        else:
            shortest = torch.vstack((shortest, _shortest))
        last_slice = cur_slice
    mu = shortest[:,1]/shortest[:,0]
    return mu

#def filter_entries(cur_vec, unused_pct=0.0):


def calc_twonn(datapts, batch_size = 64, unused_pct = 0.0, device='cpu'):
    num_pts = datapts.shape[0]
    idxs = torch.arange(num_pts, device=device)
    #mus = torch.zeros(num_pts, dtype=datapts.dtype, device=device)
    dists = torch.nn.functional.pdist(datapts, p=2)
    #mus[i] = get_mu_i_old(datapts[i], datapts.index_select(batch_dim, idxs[idxs!= i]), batch_size = batch_size)
    mus = get_mus(idxs, num_pts, dists, batch_size = batch_size, device =device).sort(descending=False).values
    mus = mus[mus.isfinite()]
    #mu_idxsort = torch.argsort(mus, descending=False)
    #p_emp = torch.zeros(num_pts, dtype=mus.dtype,device=device)
    p_emp = (torch.arange(mus.shape[0], dtype=mus.dtype,device=mus.device))/num_pts
    #print(dists.device, mus.device, p_emp.device)
    # map i/N to to proper mu indices
    #p_emp[mu_idxsort] = p_emp_unsort
    #log_mus = torch.log(mus.index_select(0, idxs[idxs != max_idx])).unsqueeze(1)
    #log_emp = -1. * torch.log(1. - p_emp.index_select(0, idxs[idxs != max_idx])).unsqueeze(1)
    #log_mus = torch.log(mus.index_select(0, idxs)).unsqueeze(1)
    #log_emp = -1. * torch.log(1. - p_emp.index_select(0, idxs)).unsqueeze(1)
    log_mus = None
    log_emp = None
    if unused_pct > 0.:
        use_idxs = int(num_pts * (1. - unused_pct))
        #log_mus = torch.log(mus.index_select(0, mu_idxsort[:use_idxs])).unsqueeze(1)
        #log_emp = -1. * torch.log(1. - p_emp.index_select(0, mu_idxsort[:use_idxs])).unsqueeze(1)
        #log_mus = torch.log(mus[mu_idxsort[:use_idxs]])
        #log_emp = -1. * torch.log(1. - p_emp[mu_idxsort[:use_idxs]])

        log_mus = torch.log(mus[:use_idxs])
        log_emp = -1. * torch.log(1. - p_emp[:use_idxs])
    else:
        #log_mus = torch.log(mus).unsqueeze(1)
        #log_emp = -1. * torch.log(1. - p_emp).unsqueeze(1)
        log_mus = torch.log(mus)
        log_emp = -1. * torch.log(1. - p_emp)
    #xtx = log_mus.T @ log_mus
    #xty = log_mus.T @ log_emp
    xtx = torch.sum(torch.pow(log_mus, 2.))
    xty = torch.sum(torch.mul(log_mus, log_emp))
    # least squares solution
    slope = xty/xtx
    return slope

