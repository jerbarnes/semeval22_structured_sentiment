from os import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from preprocessing import MyDataset, Glove, External
from padded_collate import padded_collate
from model import BiLSTMModel
#import DependencyDecoders as dd
import numpy as np
import scorer as sc
import json

#from uniparse.decoders import cle

class ModelInteractor:
    """Responsible for training the model and using it to make predictions"""

    @staticmethod
    def factory(settings, vocabs):
        if settings.unfactorized:
            return ModelInteractorUnfactorized(settings, vocabs)
        else:
            return ModelInteractorfactorized(settings, vocabs)

    def __init__(self, settings, vocabs):
        self.vocabs = vocabs
        if not settings.disable_external:
            self.external = External(settings.external)
        else:
            self.external = External(None)
        self.model = None
        self.optimizer = None
        self.train_data = None
        self.test_data = None
        self.epoch_offset = 0
        self.settings = settings

        if settings.tree:
            #self.dec = dd.DependencyDecoder()
            self.dec = True
        else:
            self.dec = None

        # which targets to take
        self.ot = settings.ot
        self.pt = settings.pt

        self.device = settings.device

        self.loss_interpolation = settings.loss_interpolation
        self.model_interpolation = settings.model_interpolation
        self.batch_size = settings.batch_size

        self.model = BiLSTMModel(self.vocabs, self.external,
                                 settings)
        self.model = self.model.to(self.settings.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=(settings.beta1, settings.beta2),
            weight_decay=settings.l2)
        self._store_settings()

    def _store_settings(self):
        with open(self.settings.dir + "settings.json", "w") as fh:
            json.dump({k: v for k,v in self.settings.__dict__.items() if k not in "device".split()}, fh)
            #for key, val in self.settings.__dict__.items():
            #    print("{}: {}".format(key,val), file=fw)

    def upd_from_other(self, other, *args):
        other_dict = other.model.state_dict()
        print(other_dict.keys())
        model_dict = self.model.state_dict()
        od = {}
        for k,v in other_dict.items():
            for a in args:
                if k.startswith(a):
                    od[k] = v
        #other_dict = {k: v for k, v in other_dict.items() if k in args}
        # 2. overwrite entries in the existing state dict
        print(od.keys())
        model_dict.update(od)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

    def freeze_params(self, *freeze):
        froze = []
        for name, param in self.model.named_parameters():
            for f in freeze:
                if name.startswith(f):
                    froze.append(name)
                    param.requires_grad = False
        print(f"froze {froze} parameters")

    def _init_training_data(self, train_path):
        self.train_data = MyDataset(
            train_path,
            vocabs=self.vocabs,
            external=self.external,
            settings=self.settings,
            elmo=self.settings.elmo_train,
            vec_dim=self.settings.vec_dim)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=padded_collate)

    def _init_test_data(self, test_path, elmo_path=None):
        self.test_data = MyDataset(
            test_path,
            vocabs=self.vocabs,
            external=self.external,
            settings=self.settings,
            elmo=elmo_path,
            vec_dim=self.settings.vec_dim)
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=padded_collate)

    def _run_train_batch(self, batch, optimizer, gradient_clipping=True):
        raise NotImplementedError()

    def _run_train_epoch(self,
                         data,
                         epoch,
                         verbose=True,
                         gradient_clipping=True):
        self.model.train()
        print_every = int(len(data) / 100) + 1

        total_loss = 0
        sequences_trained = 0

        debug_loss = []
        debug_timer = time.time()
        for i, batch in enumerate(data):
            batch.to(self.device)
            loss = self._run_train_batch(batch, self.optimizer,
                                         gradient_clipping)
            debug_loss.append(loss)
            if torch.cuda.is_available():
               print(torch.cuda.memory_allocated(self.device)/10**6)
               print(torch.cuda.memory_cached(self.device)/10**6)
               torch.cuda.empty_cache()
               print(torch.cuda.memory_cached(self.device)/10**6)

            if verbose and (i + 1) % print_every == 0:
                percentage = int((i + 1) / print_every)
                print(
                    "{}% of epoch {} ".format(percentage, epoch) +
                    "completed, current loss is {}".format(round(sum(debug_loss)/len(debug_loss), 6))
                    +
                    " averaged over the past {} sentences".format(len(debug_loss)*batch.sentence_count)
                    + " (took {} seconds)".format(round(time.time()-debug_timer, 2)), flush=True)
                debug_loss = []
                debug_timer = time.time()
            total_loss += loss
            sequences_trained += batch.sentence_count
        return total_loss, sequences_trained

    def train(self):
        settings = self.settings

        print("Training is starting for {} epochs using ".format(settings.epochs) +
              "{} with the following settings:".format(self.device))
        print()
        for key, val in settings.__dict__.items():
            print("{}: {}".format(key, val))
        print(flush=True)

        train_dataloader = self._init_training_data(settings.train)
        best_f1 = 0
        best_f1_epoch = 1 + self.epoch_offset

        for epoch in range(1 + self.epoch_offset,
                           settings.epochs + 1 + self.epoch_offset):
            start_time = time.time()
            total_loss, sequences_trained = self._run_train_epoch(
                train_dataloader, epoch, not settings.quiet,
                not settings.disable_gradient_clip)
            total_time = round(time.time() - start_time, 2)
            print("#" * 50)
            print("Epoch {}".format(epoch))
            print("loss {}".format(total_loss))
            print("execution time {}s".format(total_time) \
            + " ({} trained sequences/s)".format(round(sequences_trained/(total_time))))
            print("#" * 50, flush=True)
            if not settings.disable_val_eval:
                entries, predicted, other_predicted = self.predict(settings.val, settings.elmo_dev)
                #a,d,b,c = zip(*((entry[0], len(entry[4]), entry[1].numpy().shape, predicted[entry[0]].numpy().shape) for entry in entries))
                #print([(x,w,y,z) for x,w,y,z in zip(a,d,b,c) if y!=z])
                f1, _ = sc.score(*zip(*((entry[1][self.pt].numpy(), predicted[entry[0]].numpy()) for entry in entries)))
                print("Primary Dev F1 on epoch {} is {:.2%}".format(epoch, f1))

                if len(other_predicted) > 0:
                    other_f1, _ = sc.score(*zip(*((entry[1][self.ot].numpy(), other_predicted[entry[0]].numpy()) for entry in entries)))
                    print("Secondary Dev F1 on epoch {} is {:.2%}".format(epoch, other_f1))
                #f1 = sc.score()
                improvement = f1 > best_f1
                elapsed = epoch - best_f1_epoch
                es_active = settings.early_stopping > 0

                if (es_active and not improvement
                        and elapsed == settings.early_stopping):
                    print(
                        "Have not seen any improvement for {} epochs".format(elapsed))
                    print(
                        "Best F1 was {} seen at epoch #{}".format(best_f1, best_f1_epoch)
                    )
                    break
                else:
                    if improvement:
                        best_f1 = f1
                        best_f1_epoch = epoch
                        print("Saving {} model".format(best_f1_epoch))
                        self.save("best_model.save", epoch)
                    else:
                        print("Have not seen any improvement for {} epochs".format(elapsed))
                    print("Best F1 was {:.2%} seen at epoch #{}".format(best_f1, best_f1_epoch))

            if settings.enable_train_eval:
                entries, predicted, other_predicted = self.predict(settings.train, settings.elmo_train)
                train_f1, _ = sc.score(*zip(*((entry[1][self.pt].numpy(), predicted[entry[0]].numpy()) for entry in entries)))
                print("Sem Train F1 on epoch {} is {:.2%}".format(epoch, train_f1))

                if len(other_predicted) > 0:
                    other_train_f1, _ = sc.score(*zip(*((entry[1][self.ot].numpy(), other_predicted[entry[0]].numpy()) for entry in entries)))
                    print("Syn Train F1 on epoch {} is {:.2%}".format(epoch, other_train_f1))

            if settings.save_every:
                self.save("{}_epoch{}.save".format(int(time.time()), epoch), epoch)
            else:
                self.save("last_epoch.save", epoch)

    def _run_test_batch(self, batch):
        raise NotImplementedError()

    def _clip_grad(self, gradient_clipping):
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

    def predict(self, data_path, elmo_path=None):
        print("Predicting data from", data_path)
        test_loader = self._init_test_data(data_path, elmo_path)
        self.model.eval()
        predictions = {}
        other_predictions = {}
        for batch in test_loader:
            batch.to(self.device)
            print(".", end="")
            sys.stdout.flush()
            with torch.no_grad():
                pred, other_pred = self._run_test_batch(batch)
                predictions.update(pred)
                other_predictions.update(other_pred)
        #for k,v in predictions.items():
        #    print(k, v.shape)
        print("Done")

        #return self.test_data.data, predictions
        return self.test_data, predictions, other_predictions

    def save(self, path, epoch):
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available(
        ) else None
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "vocabs": self.vocabs,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_state,
            "epoch": epoch
        }
        torch.save(state, self.settings.dir + path)

    def load(self, path):
        print("Restoring model from {}".format(path))
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.model = self.model.to(self.settings.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.vocabs = state["vocabs"]
        torch.set_rng_state(state["rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng_state"])
        self.epoch_offset = state["epoch"]

    def other_loss(self, other_edge_scores, other_label_scores, batch, loss):
        #####
        if torch.cuda.is_available():
            print("other_loss")
            print(torch.cuda.memory_allocated(self.device)/10**6)
            print(torch.cuda.memory_cached(self.device)/10**6)
            torch.cuda.empty_cache()
            print(torch.cuda.memory_cached(self.device)/10**6)

        other_label_scores_transposed = other_label_scores.transpose(0, 1)
        other_edge_targets = (batch.targetss[self.ot] > 0)
        other_unpadded_edge_scores = other_edge_scores[batch.unpadding_mask]
        other_unpadded_edge_targets = other_edge_targets[batch.unpadding_mask]
        other_edge_loss = F.binary_cross_entropy_with_logits(
            other_unpadded_edge_scores, other_unpadded_edge_targets.float())
        other_gold_mask = other_edge_targets
        other_gold_mask_expanded = other_gold_mask.unsqueeze(0).expand_as(
            other_label_scores_transposed)
        other_gold_targets = batch.targetss[self.ot][other_gold_mask]
        if len(other_gold_targets) > 0:
            # Extract the scores for the existing labels
            other_scores = other_label_scores_transposed[other_gold_mask_expanded]
            # (labels x predictions)
            other_scores = other_scores.view(-1, len(other_gold_targets))

            # scores.t() => [#predictions x #labels], gold_target [#predictions]
            # gold_target needs to contain the indices of the correct labels.
            # Since gold_target labels are in the range 1..#labels, 1 is subtracted
            other_label_loss = F.cross_entropy(other_scores.t(),
                                         other_gold_targets - 1)

            other_loss = self.loss_interpolation * other_label_loss + (
                1 - self.loss_interpolation) * other_edge_loss
        else:
            other_loss = (1 - self.loss_interpolation) * other_edge_loss

        loss *= 1 - self.model_interpolation
        loss += other_loss * self.model_interpolation

        return loss
        #####
    def other_predict(self, other_edge_scores, other_label_scores, i, size, other_predictions, batch):
       ####
       other_unpadded_edge_scores = other_edge_scores[i, :size, :size]
       other_unpadded_label_scores = other_label_scores[i, :, :size, :size]
       other_edge_prediction = self.predict_edges(other_unpadded_edge_scores)
       other_label_prediction = self.predict_labels(other_unpadded_label_scores)#.cpu().numpy()
       other_combined_prediction = (other_edge_prediction * other_label_prediction)
       other_predictions[batch.graph_ids[i]] = other_combined_prediction.cpu()
       ####


class ModelInteractorfactorized(ModelInteractor):
    def __init__(self, settings, vocabs):
        return super().__init__(settings, vocabs)

    def predict_edges(self, scores):
        #if self.dec is not None:
        #    prediction = self.dec.parse_nonproj(scores.cpu().numpy().astype(np.float64))
        #    print(prediction)
        #    return prediction
        #print((scores >= 0).cpu().numpy())
        #return (scores >= 0).cpu().numpy()
        return (scores >= 0).float()

    def predict_labels(self, scores):
        # +1 since argmax will predict dimension in 0..n-1, while labels are in range 1..n
        return torch.argmax(scores, dim=0).float() + 1

    def _run_train_batch(self, batch, optimizer, gradient_clipping=True):
        optimizer.zero_grad()
        #print("seqlengths", batch.seq_lengths)
        # edge [batch x head_seq x dependent_seq]
        # label [batch x label x head_seq x dependent_seq]
        other_edge_scores, other_label_scores, edge_scores, label_scores = self.model(batch.targetss, batch.seq_lengths, batch.chars, *batch.indices)

        # (label x batch x head_seq x dependent_seq)
        label_scores_transposed = label_scores.transpose(0, 1)


        # TODO batch.targets is list of targets
        edge_targets = (batch.targetss[self.pt] > 0)

        unpadded_edge_scores = edge_scores[batch.unpadding_mask]
        unpadded_edge_targets = edge_targets[batch.unpadding_mask]


        edge_loss = F.binary_cross_entropy_with_logits(
            unpadded_edge_scores, unpadded_edge_targets.float())


        # Masks filtering out gold dependencies
        gold_mask = edge_targets
        gold_mask_expanded = gold_mask.unsqueeze(0).expand_as(
            label_scores_transposed)


        # Extract only the gold labels
        # TODO batch.targets is list of targets
        gold_targets = batch.targetss[self.pt][gold_mask]


        # Only caculate label loss if there actually are any gold labels
        if len(gold_targets) > 0:
            # Extract the scores for the existing labels
            scores = label_scores_transposed[gold_mask_expanded]
            # (labels x predictions)
            scores = scores.view(-1, len(gold_targets))

            # scores.t() => [#predictions x #labels], gold_target [#predictions]
            # gold_target needs to contain the indices of the correct labels.
            # Since gold_target labels are in the range 1..#labels, 1 is subtracted
            label_loss = F.cross_entropy(scores.t(),
                                         gold_targets - 1)

            loss = self.loss_interpolation * label_loss + (
                1 - self.loss_interpolation) * edge_loss
        else:
            loss = (1 - self.loss_interpolation) * edge_loss

        if other_edge_scores is not None:
            loss = self.other_loss(other_edge_scores, other_label_scores, batch, loss)

        ret_loss = float(loss.detach())

        loss.backward()
        self._clip_grad(gradient_clipping)
        optimizer.step()

        return ret_loss

    def _run_test_batch(self, batch):
        # edge [batch x head_seq x dependent_seq]
        # label [batch x label x head_seq x dependent_seq]
        other_edge_scores, other_label_scores, edge_scores, label_scores = self.model(batch.targetss, batch.seq_lengths, batch.chars,
                                               *batch.indices)

        predictions = {}
        other_predictions = {}

        for i, size in enumerate(batch.seq_lengths):
            size = size.item()
            unpadded_edge_scores = edge_scores[i, :size, :size]
            unpadded_label_scores = label_scores[i, :, :size, :size]

            edge_prediction = self.predict_edges(unpadded_edge_scores)
            label_prediction = self.predict_labels(unpadded_label_scores)#.cpu().numpy()
            #print(type(edge_prediction), type(label_prediction))
            combined_prediction = (edge_prediction * label_prediction)

            predictions[batch.graph_ids[i]] = combined_prediction.cpu()

            if other_edge_scores is not None:
                self.other_predict( other_edge_scores, other_label_scores, i, size, other_predictions, batch)

        return predictions, other_predictions


class ModelInteractorUnfactorized(ModelInteractor):
    def __init__(self, settings, vocabs):
        return super().__init__(settings, vocabs)

    def _run_train_batch(self, batch, optimizer, gradient_clipping=True):
        optimizer.zero_grad()

        #print("seqlengths", batch.seq_lengths)
        # label [batch x label x head_seq x dependent_seq]
        other_edge_scores, other_label_scores, _, label_scores = self.model(batch.targetss, batch.seq_lengths, batch.chars, *batch.indices)

        # (label x batch x head_seq x dependent_seq)
        label_scores_transposed = label_scores.transpose(0, 1)

        # Unpad scores
        scores = label_scores_transposed[batch.unpadding_mask.unsqueeze(0).
                                         expand_as(label_scores_transposed)]
        # (labels x predictions)
        scores = scores.view(label_scores_transposed.size(0), -1)
        # TODO batch.targets is list of targets
        gold_targets = batch.targetss[self.pt][batch.unpadding_mask]
        # scores.t() => [#predictions x #labels], gold_target [#predictions]
        # gold_target needs to contain the indices of the correct labels.
        # TODO which loss do you want? multi-margin does not work yet (maybe)
        loss = F.cross_entropy(scores.t(), gold_targets)
        #loss = F.multi_margin_loss(scores.t(), gold_targets)

        if other_edge_scores is not None:
            loss = self.other_loss(other_edge_scores, other_label_scores, batch, loss)

        ret_loss = float(loss)

        loss.backward()
        self._clip_grad(gradient_clipping)
        optimizer.step()

        return ret_loss

    def _run_test_batch(self, batch):
        # [batch x label x head_seq x dependent_seq]
        other_edge_scores, other_label_scores, _, label_scores = self.model(batch.targetss, batch.seq_lengths, batch.chars,
                                      *batch.indices)
        predictions = {}
        other_predictions = {}

        for i, size in enumerate(batch.seq_lengths):
            size = size.item()
            scores = label_scores[i, :, :size, :size]

            prediction = torch.argmax(scores, dim=0).float()
            predictions[batch.graph_ids[i]] = prediction.cpu()
            #print(batch.graph_ids[i], size, i, scores.shape, prediction.cpu().shape, predictions[batch.graph_ids[i]].shape)
            if other_edge_scores is not None:
                #print(batch.graph_ids[i], prediction.shape)
                #print(prediction)
                self.other_predict( other_edge_scores, other_label_scores, i, size, other_predictions, batch)

        #print(predictions[batch.graph_ids[0]])
        #print(predictions[batch.graph_ids[i]])
        return predictions, other_predictions


    # predict edges/labels added for other_predict
    def predict_edges(self, scores):
        if self.dec is not None:
            #prediction = self.dec.parse_nonproj(scores.cpu().numpy().astype(np.float64))
            heads = cle.parse_nonproj(scores.cpu().numpy().astype(np.float64))
            prediction = np.zeros(scores.shape, int)
            # NOTE there is a possibility for multiple roots...
            # use the first root as root and the others as its children
            # root_found
            root = -1
            for m,h in enumerate(heads[1:]):
                if root == -1 and h == 0:
                    root = m+1
                elif root != -1 and h == 0:
                    h = root
                prediction[h,m+1] = 1
            #prediction = self.dec.parse_proj(scores.cpu().numpy().astype(np.float64))
            #print(torch.Tensor(prediction))
            #print((scores >= 0).float())
            return torch.Tensor(prediction)
        return (scores >= 0).float()

    def predict_labels(self, scores):
        # +1 since argmax will predict dimension in 0..n-1, while labels are in range 1..n
        return torch.argmax(scores, dim=0).float() + 1

