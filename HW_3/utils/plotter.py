import matplotlib.pyplot as pl


class ErrorPlotter:

    enabled = False

    epochs = []

    errors = []

    def __init__(self) -> None:
        super().__init__()
        if self.enabled:
            pl.title = "Error for every training cycle"
            pl.xlabel = "Epoch"
            pl.ylabel = "Error"
            pl.ion()
            pl.show()

    def append_error(self, error, batch_size):
        if self.enabled:
            epochs = len(self.epochs)
            error = error[0][0]
            pl.title = f"Error = {error}"
            self.epochs.append(epochs + 1)
            self.errors.append(error)
            pl.xlim(len(self.errors) - batch_size, len(self.errors))
            n = 1
            if epochs == 0 or epochs % 100 == 0:
                pl.plot(self.epochs[::n], self.errors[::n])
                pl.draw()
                pl.pause(.001)
