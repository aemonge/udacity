import time

def print_step(epoch, epochs, start_time, time_count,
               tr_loss, vd_loss, vd_correct,
               validate_loader_count, train_loader_count):

  training_loss = tr_loss / train_loader_count
  validate_loss = vd_loss / validate_loader_count

  time_now = time.time()
  time_count += time_now - start_time
  time_f = time.strftime("%H:%M:%S", time.gmtime(time_now - start_time))

  print('\033[F\033[K', end='\r') # back prev line and clear
  print(f"  Epoch: {epoch + 1}/{epochs}\n",
        f"    Losses [ training: {training_loss:.4f}, validate: {validate_loss:.4f} ],\n",
        f"    Accuracy: {(vd_correct * 100 / validate_loader_count):.2f}%",
        f" Time: {time_f}\n"
  )

  return time_now
