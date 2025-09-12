# Add functions or classes used for data loading and preprocessing

class NTAvSGD:
    def __init__(self, params, lr=30, n=5, L=1000):
        self.params = list(params)
        self.lr = lr
        self.n = n  # Non-monotone interval
        self.L = L  # Logging interval (iterations per evaluation)
        self.optimizer = torch.optim.SGD(self.params, lr=lr)
        
        # Tracking variables
        self.iteration = 0
        self.best_val_loss = float('inf')
        self.val_losses = []
        self.weights = []
        self.averaging = False
        self.trigger_count = 0
        
    def step(self, closure=None):
        self.optimizer.step(closure)
        self.iteration += 1
        
        # Store weights if we're in the averaging phase
        if self.averaging:
            self.weights.append([p.data.clone() for p in self.params])
        
    def update_val_loss(self, val_loss):
        """Call this after each validation evaluation"""
        self.val_losses.append(val_loss)
        
        # Check if we should start averaging
        if not self.averaging:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.trigger_count = 0
            else:
                self.trigger_count += 1
                
            # Non-monotonic trigger condition
            if self.trigger_count >= self.n:
                self.start_averaging()
                
    def start_averaging(self):
        """Begin weight averaging"""
        self.averaging = True
        self.weights = []
        
    def get_averaged_weights(self):
        """Returns averaged weights if averaging has been triggered"""
        if not self.averaging or len(self.weights) == 0:
            return None
            
        # Average all stored weights
        avg_weights = []
        for i in range(len(self.weights[0])):
            avg = torch.stack([w[i] for w in self.weights]).mean(0)
            avg_weights.append(avg)
            
        return avg_weights