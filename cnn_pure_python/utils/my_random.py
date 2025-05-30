class MyPseudoRandom:
    def __init__(self, m: int = 7, a: int = 3, c: int = 3, seed: int = 5) -> None:
        """
        m: modulus
        a: multiplier
        c: increment
        X_init: seed, start value
        return pseudo random
        """
        self.m = m
        self.a = a
        self.c = c
        self.X = seed
        
    def next(self) -> float:
        self.X = (self.a * self.X + self.c) % self.m
        return self.X / self.m

    def generate_randn(self, *shape):
        if not shape:
            return self.next()
        elif len(shape) == 1:
            return [self.next()/10 for _ in range(shape[0])]
        else:
            inner_list = self.generate_randn(*shape[1:])
            return [inner_list for _ in range(shape[0])]

