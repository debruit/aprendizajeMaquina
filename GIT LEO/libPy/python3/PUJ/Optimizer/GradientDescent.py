# =========================================================================
# @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
# =========================================================================

import numpy
from .Base import *


class GradientDescent(Base):

    '''
    '''

    def __init__(self, cost):
        super().__init__(cost)
    # end def

    def Fit(self):
        # Initial values
        [J0, g] = self.evaluate( 0 )
        start_batch = 1
        J1=0

        # Prepare loop
        stop = False
        self.m_Iteration = 0


        # Main loop
        while not stop:

            # Advance
            for b in range(start_batch, self.m_Cost.numberOfBatches()):
                self.m_Cost.updateModel(-self.m_Alpha * g)
                [J1, g] = self.evaluate(b)
            # end for
            start_batch = 0
            self.m_Cost.shuffle()
            if not self.m_DebugFunction is None:
                stop = self.m_DebugFunction(
                    self.m_Cost.model(), self.m_Iteration, J0, J0 - J1,
                    self.m_Iteration % self.m_NumberOfDebugIterations == 0
                )
            # end if

            # Update iterations
            self.m_Iteration += 1
            stop = stop or self.m_Iteration >= self.m_NumberOfIterations
            stop = stop or (abs(J0 - J1) < self.m_Epsilon)

            # Final debug call
            if not self.m_DebugFunction is None and stop:
                self.m_DebugFunction(
                    self.m_Cost.model(), self.m_Iteration - 1, J0, J0 - J1, True
                )
            # end if
            J0 = J1
    # end while
    # end def

    # def Fit(self):
    #     [J0, g] = self.m_Cost.evaluate(True)
    #     [Jr, gr] = self.regularize()
    #     J0 += Jr
    #     g += gr
    #     stop = False
    #     self.m_RealIterations = 0
    #     while not stop:
    #         self.m_Cost.updateModel(-self.m_Alpha * g)
    #         [J1, g] = self.m_Cost.evaluate(True)
    #         [Jr, gr] = self.regularize()
    #         J1 += Jr
    #         g += gr
    #         if not self.m_DebugFunction is None:
    #             stop = self.m_DebugFunction(self.m_Cost.model(
    #             ), self.m_RealIterations, J1, J0 - J1, self.m_RealIterations % self.m_NumberOfDebugIterations == 0)
    #         # end if
    #         stop = stop or self.m_RealIterations >= self.m_NumberOfIterations
    #         stop = stop or ((J0 - J1) < self.m_Epsilon)
    #         self.m_RealIterations += 1
    #         J0 = J1
    #     # end while
    # # end def

# eof - $RCSfile$
