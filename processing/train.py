
import numpy as np
import torch
import time
import random
import torch.nn.functional as F
#from Masks import generate_square_subsequent_mask
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=3)


def calculate_synthetic_loss(probs=None,
                             is_next_pred=None,
                             targets=None,
                             is_next=None,
                             prob_weight=None,
                             sum_loss=True):
    """
    Calculate loss
    """
    # Find loss for missing vectors
    is_next_preds = is_next_pred.view(is_next.shape)
    is_next_sequence = F.binary_cross_entropy_with_logits(is_next_preds,
                                                          is_next,
                                                          reduction='mean')

    is_next_acc_preds = torch.round(torch.sigmoid(is_next_pred))
    is_next_acc_preds = is_next_acc_preds.view(is_next.shape)
    is_next_acc = (is_next_acc_preds == is_next).sum().float()
    is_next_acc = is_next_acc / (is_next.shape[0])

    # Find loss for altered vectors
    altered_preds = probs.view(targets.shape)
    altered_vectors = F.binary_cross_entropy_with_logits(
        altered_preds, targets.cuda())

    altered_vectors_acc = torch.round(torch.sigmoid(probs))
    altered_vectors_acc = altered_vectors_acc.view(targets.shape)
    altered_vectors_acc = (altered_vectors_acc == targets).sum().float()
    altered_vectors_acc = altered_vectors_acc / \
        (targets.shape[0]*targets.shape[1])

    if not sum_loss:
        loss = ((prob_weight * altered_vectors) +
                ((1 - prob_weight) * is_next_sequence))
    else:
        loss = is_next_sequence + altered_vectors
    return loss, is_next_sequence.item(), altered_vectors.item(
    ), is_next_acc.item() * 100, altered_vectors_acc.item() * 100




def calculate_classification_loss_and_accuracy(preds,targets,pos_weight=None):
    preds = preds.view(targets.shape)

    if pos_weight:
        loss = F.mse_loss(preds,targets,reduction='mean',pos_weight=pos_weight)
    else:
        loss = F.mse_loss(preds,targets,reduction='mean')
    
    # Date weighting
    preds = preds.detach().cpu().numpy()
    return loss


def train_model_classification(train_dataloader,
                        num_epochs,
                        model=None,
                        optim=None,
                        pos_weight=None):

    total_loss = 0
    i = 0
    n_iter = 0

    for epoch in range(num_epochs):

        epoch_loss = 0
        epoch_f1 = 0

        i += 1
        j = 0

        for batch in train_dataloader:
            j += 1

            src = batch['encoder'].cuda()
            dec = batch['decoder'].cuda()
            targets = batch['target'].cuda()

            dec_mask = None
            src_mask = None

            preds = model(src.float(), dec.float(), src_mask, dec_mask)
            preds = preds

            optim.optimizer.zero_grad()
            loss, f1 = calculate_classification_loss_and_accuracy(preds=preds,
                                              targets=targets,
                                              pos_weight=pos_weight)

            loss.backward()

            optim.optimizer.step()
            optim.step()

            epoch_loss += loss.item()
            epoch_f1 += f1
            total_loss += loss.item()


            n_iter += 1

def temporal_proximity_weight(x_date, y_date, alpha=0.5):
    """
    Temporal proximity weight
    """
    pass


def train_model_synthetic(train_dataloader,
                          val_dataloader,
                          num_epochs,
                          model=None,
                          optim=None,
                          val_int=10,
                          save_as="",
                          mask_rate=0.25,
                          prob_weight=0.9,
                          mean=0,
                          std=1,
                          sum_loss=False,
                          val_writer=None):

    """
    
    
    @Author : Håkon Måløy
    """
    best_val = np.inf

    total_loss = 0
    i = 0
    n_iter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_val_loss = 0
        epoch_next = 0
        epoch_next_acc = 0
        epoch_altered_acc = 0
        epoch_altered_loss = 0
        i += 1
        t1 = time.time()
        j = 0
        n_vals = 0

        val_iter = 0
        for batch in train_dataloader:
            j += 1
            # Get data
            src = batch['encoder']
            dec = batch['decoder']
            wrng_seq = batch['target']

            # generate_square_subsequent_mask(dec.shape[1]).cuda()
            dec_mask = None
            src_mask = None

            # Randomize the next sequence for is_next_sequence prediction
            dec, wrng_seq, is_next = randomize_next_sequence(dec, wrng_seq)

            # Set random x% of data to zero vector
            src, dec, targets = create_masked_LM_vectors(
                mask_rate, src, dec, wrng_seq)

            transformer_out, probs, is_next_pred = model(
                src.float(), dec.float(), src_mask, dec_mask)

            optim.optimizer.zero_grad()
            loss, next_seq, altered_loss, is_next_acc, altered_acc = calculate_synthetic_loss(
                probs=probs,
                is_next_pred=is_next_pred,
                targets=targets,
                is_next=is_next,
                prob_weight=prob_weight,
                sum_loss=sum_loss)
            loss.backward()
            optim.optimizer.step()
            optim.step()

            epoch_loss += loss.item()
            epoch_next += next_seq
            epoch_next_acc += is_next_acc
            epoch_altered_acc += altered_acc
            epoch_altered_loss += altered_loss
            total_loss += loss.item()

            n_iter += 1

            t2 = time.time()
            total_time = (t2 - t1)

        print(
            'Epoch: %i, loss = %.3f,  next_seq_loss: %.3f, next_seq_acc: %.3f, altered_loss: %.3f, altered_acc: %.3f, Time: %.2f'
            % (i, (epoch_loss / j), (epoch_next / j), (epoch_next_acc / j),
               (epoch_altered_loss / j), (epoch_altered_acc / j), total_time))

        epoch_val_loss = epoch_val_loss / val_iter
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            print("Saving model to: " + save_as)
            torch.save(model.state_dict(), save_as + '_state_dict')
            torch.save(model, save_as)

    return n_iter


def get_n_targets(mask_rate, seq_length):
    """
    

    @Author : Håkon Måløy
    """
    # The number of targets should be an int
    n_targets = int(mask_rate * seq_length)
    return n_targets


def create_masked_LM_vectors(mask_rate, src, dec, wrng_seq):
    """
    Replaces vectors in the input tensor to make a task of determining which indexes have been changed.
    @Author : Håkon Måløy
    """
    # Do LM masking for each of the inputs
    src, src_trg = do_LM_masking(mask_rate, src, wrng_seq, dec)
    dec, dec_trg = do_LM_masking(mask_rate, dec, wrng_seq, src)

    # Concatenate the targets to create one big target vector
    masked_targets = torch.cat((src_trg, dec_trg), dim=1)

    return src.cuda(), dec.cuda(), masked_targets.cuda()


def do_LM_masking(mask_rate, tensor_to_mask, wrng_seq, src):
    """
    
    
    @Author : Håkon Måløy
    """
    number_of_targets = get_n_targets(mask_rate, tensor_to_mask.shape[1])
    # Make a target tensor
    targets = torch.zeros(tensor_to_mask.shape[0], tensor_to_mask.shape[1])
    #zero_vector = torch.zeros(tensor_to_mask.shape[2])

    # Go over input tensor
    for batch in range(tensor_to_mask.shape[0] - 1):
        # Sample n indexes
        indexes = random.sample(list(range(0, tensor_to_mask.shape[1] - 1)),
                                number_of_targets)
        for index in indexes:
            # A random number
            rand = random.random()

            # Select a random index from the wrong tensor
            random_batch_select = random.randint(0,
                                                 tensor_to_mask.shape[0] - 1)
            random_index_in_sequence = random.randint(
                0, tensor_to_mask.shape[1] - 1)

            # Change the vector to a vector from the src_sequence 80% of the time
            if rand <= 1.0:
                # Change the vector to the random vector 20% of the time
                mask_vector = wrng_seq[random_batch_select,
                                       random_index_in_sequence]
            else:
                # Change vector to vector from src 80% of the time
                mask_vector = src[batch, random_index_in_sequence]

            # Replace the tensor with our selected replacement tensor
            tensor_to_mask[batch, index] = mask_vector

            # Set this index to one in our target
            targets[batch, index] = 1.0
    return tensor_to_mask, targets


def randomize_next_sequence(dec, wrng_seq, prob=0.5):
    """
    With a given probability, a sequence will be exhanged for a sequence not actually following the previous sequence. The is_next variable is then also set to false. This creates the is_next sequence dataset.
    @Author : Håkon Måløy
    """
    is_next = torch.ones(dec.shape[0])
    for i in range(dec.shape[0]):
        if random.random() < prob:
            # Find random index
            # print(wrng_seq.shape[0])
            idx = random.randint(0, wrng_seq.shape[0] - 1)
            # Store the sequence from dec
            #tmp = dec[i]
            # Replace sequence of dec with a random one
            dec[i] = wrng_seq[idx]
            # Set the
            #wrng_seq[idx] = tmp
            is_next[i] = 0.0
    return dec, wrng_seq, is_next.cuda()


if __name__ == "__main__":
    