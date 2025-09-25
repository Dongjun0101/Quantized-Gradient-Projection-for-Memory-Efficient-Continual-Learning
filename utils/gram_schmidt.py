import torch
import numpy as np

def gram_schmidt(bases):
    """
    Perform Gram-Schmidt orthogonalization on a matrix of basis vectors.
    
    Args:
        bases (torch.Tensor): A matrix where each column is a basis vector.
        
    Returns:
        torch.Tensor: A matrix where each column is an orthonormal vector.
    """
    num_vectors = bases.size(1)
    orthogonalized = torch.zeros_like(bases)
    
    for i in range(num_vectors):
        # Start with the original basis vector
        vec = bases[:, i]
        
        # Subtract projections onto the previous orthogonal vectors
        for j in range(i):
            proj = torch.dot(orthogonalized[:, j], vec) / torch.dot(orthogonalized[:, j], orthogonalized[:, j])
            vec = vec - proj * orthogonalized[:, j]
        
        # Normalize the vector
        norm = torch.norm(vec)
        if norm > 1e-8:  # Prevent division by zero
            orthogonalized[:, i] = vec / norm
    
    return orthogonalized

def update_GPM(mat_list, threshold, orth_basis=[], feature_list=[], compress=False, rank=0, device=None):
    if(rank==0):
        print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])

            if compress:
                f_shape= np.shape(feature_list[i])
                M_MT =np.dot(feature_list[i],feature_list[i].transpose())
                I= np.identity(f_shape[0])
                Uo,So,Vo= np.linalg.svd(I-M_MT)
                orth_basis.append(Uo[:,0:f_shape[0]-f_shape[1]])

    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation 
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)

            # criteria
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui

            if compress:
                f_shape= np.shape(feature_list[i])
                M_MT =np.dot(feature_list[i],feature_list[i].transpose())
                I= np.identity(f_shape[0])
                Uo,So,Vo= np.linalg.svd(I-M_MT)
                orth_basis[i]= Uo[:,0:f_shape[0]-f_shape[1]]

    if(rank==0):
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(feature_list)):
            print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
            if compress:
                print ('Orth Basis Layer {} : {}/{}'.format(i+1,orth_basis[i].shape[1], orth_basis[i].shape[0]))
        print('-'*40)
    return feature_list, orth_basis

def update_agg_GPM(feature_agg_list, feature_mine_list, threshold, rank):

    for i in range(len(feature_mine_list)):
        activation = feature_agg_list[i]
        U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
        sval_total = (S1**2).sum()
        # Projected Representation 
        act_hat = activation - np.dot(np.dot(feature_mine_list[i],feature_mine_list[i].transpose()), activation)
        U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)

        # criteria
        sval_hat = (S**2).sum()
        sval_ratio = (S**2)/sval_total               
        accumulated_sval = (sval_total-sval_hat)/sval_total
        
        r = 0
        for ii in range (sval_ratio.shape[0]):
            if accumulated_sval < threshold[i]:
                accumulated_sval += sval_ratio[ii]
                r += 1
            else:
                break
        if r == 0:
            print ('Skip Updating GPM for layer: {}'.format(i+1)) 
            continue
        
        # update GPM
        Ui=torch.cat((feature_mine_list[i],U[:,0:r]), dim=1)
        if Ui.shape[1] > Ui.shape[0] :
            feature_mine_list[i]=Ui[:,0:Ui.shape[0]]
        else:
            feature_mine_list[i]=Ui

    if(rank==0):
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(feature_mine_list)):
            print ('Layer {} : {}/{}'.format(i+1,feature_mine_list[i].shape[1], feature_mine_list[i].shape[0]))
        print('-'*40)
        
    return feature_mine_list