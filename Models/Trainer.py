import torch
import time
import sys
import sklearn.metrics

class Trainer:
    
    def __init__(self, model, train_data, val_data, opt, loss, file_name=""):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_func = loss
        self.optimizer = opt
        self.file_name = file_name

    def accuracy(self, out, labels):
        return (torch.argmax(out, dim=1)==labels).float().mean() * 100

    def train(self, epochs=100, batch_size=64):
        print("Starting training...")
        num_batches = self.train_data.get_batches_in_epoch(batch_size)
        f = open(self.file_name + "_TrainingProgress" + str(time.time()) + ".txt", "w")

        best_acc = 0
        best_auroc = 0

        for e in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            for b in range(num_batches):
                text, ims, labels, lengths = self.train_data.get_batch(batch_size, b)
                pred = self.model(text, ims, lengths)
                loss = self.loss_func(pred, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                print("Batch " + str(b+1) + "/" + str(num_batches) + ", Epoch " + str(e+1) + "/" + str(epochs))
                print("Curr loss: " + str(loss.item()))

                accuracy = self.accuracy(pred, labels)
                print("Curr Accuracy: " + str(accuracy))

                sys.stdout.write("\033[F") # Cursor up one line
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[F")

                epoch_loss += loss.item()
                epoch_accuracy += accuracy
            
            print("Epoch complete!")
            f.write(str(e))
            f.write("\n")

            avg_loss = epoch_loss / num_batches
            f.write("Train loss: ")
            f.write("\n")
            f.write(str(avg_loss))
            f.write("\n")
            print("Avg loss: " + str(avg_loss))

            avg_acc = epoch_accuracy / num_batches
            f.write("Train acc: ")
            f.write("\n")
            f.write(str(avg_acc))
            f.write("\n")
            print("Avg acc: " + str(avg_acc))

            

            with torch.no_grad():
                #self.model.eval()
                val_batches = self.val_data.get_batches_in_epoch(batch_size)
                eval_loss = 0
                eval_acc = 0
                for b in range(val_batches):
                    text, ims, labels, lengths = self.val_data.get_batch(batch_size, b)
                    pred = self.model(text, ims, lengths)

                    loss = self.loss_func(pred, labels)
                    eval_loss += loss.item()

                    eval_acc += self.accuracy(pred, labels)

                eval_loss = eval_loss  / val_batches
                eval_acc = eval_acc / val_batches
                eval_acc = eval_acc.item()

                f.write("Eval loss: ")
                f.write("\n")
                f.write(str(eval_loss))  
                f.write("\n")    
                f.write("Eval acc: ")
                f.write("\n")
                f.write(str(eval_acc))
                f.write("\n")

                auroc = self.auroc()
                f.write("Eval auroc: ")
                f.write("\n")
                f.write(str(auroc))
                f.write("\n")

                print("Eval loss: " + str(eval_loss))
                print("Eval acc: " + str(eval_acc))

                if eval_acc > best_acc or auroc > best_auroc:
                    model_path = self.file_name + "_model_" + str(eval_acc) + "_" + str(auroc) + ".pt"
                    torch.save(self.model, model_path)
                    best_acc = eval_acc
                    best_auroc = auroc

            #self.model.train()
            
            self.train_data.shuffle()

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            f.write("\n")
            f.write("\n")

    def auroc(self):
        f = open(self.file_name + "_TrainingProgress" + str(time.time()) + ".txt", "w")
        with torch.no_grad():
                #self.model.eval()
                val_batches = self.val_data.get_batches_in_epoch(16)
                true_labels = []
                predictions = []


                
                for b in range(val_batches):
                    text, ims, labels, lengths = self.val_data.get_batch(16, b)
                    pred = self.model(text, ims, lengths)
                    pred = torch.nn.functional.softmax(pred, dim=1)

                    true_labels += labels.tolist()

                    predictions += pred[:, 1].tolist()

                #self.model.train()

                
                fpr, tpr, _ = sklearn.metrics.roc_curve(y_true = true_labels, y_score = predictions, pos_label = 1)
                return sklearn.metrics.auc(fpr, tpr)
                     
