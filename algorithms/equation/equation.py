class Equation:
    def __init__(self, a, b, c):
        """
        Structured class to store the main information about each delivery.

        :param a: first parameter
        :param b: second parameter
        :param c: third parameter

        """

        self.a = a
        self.b = b
        self.c = c

    def __str__(self):
        return "( %.2fx² + %.2fx + %.2f)" \
               % (self.a, self.b, self.c)



