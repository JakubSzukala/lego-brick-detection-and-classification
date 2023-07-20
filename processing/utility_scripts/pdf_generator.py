from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skewnorm 

class PDFFitter:
    """
    https://machinelearningmastery.com/probability-density-estimation/
    
    Used pdf:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html

    Skewness is "tail" of the distribution. Negative skew will have wider range 
    of values to the left of the mean (hill to the right), positive skew will 
    have wider range of values to the right (hill to the left).
    
    Fitting:
    https://stackoverflow.com/questions/50140371/scipy-skewnormal-fitting

    ppf - ppf(0.1) returns a value of random variable, for which draw will fall
    10 percent of times / inverse of cdf (cdf gives probability and this gives
    value given prob as argument)
    """
    def __init__(self, data_filepath, designator, column, epsilon):
        self.df = pd.read_csv(data_filepath)
        self.data_column = column
        self.designator = designator
        self.epsilon = epsilon
                
        # Calculate parameters of distribution
        self.a, self.loc, self.scale = skewnorm.fit(self.df[self.data_column]) # type: ignore
        
        # Create frozen object for calculated dist
        self.frozen_dist = skewnorm(self.a, self.loc, self.scale)
    

    def print_info(self):
        print('Params for {}: '.format(self.data_column))
        print('Skew: ', self.a, 'Mean / Loc: ', self.loc, 'Scale: ', self.scale)

    
    def plot_pdf_and_hist(self, bin_start, bin_end, bin_step):
        bins = [b for b in np.arange(bin_start, bin_end, bin_step)]
        fig, ax = plt.subplots(1, 1)
        ax.hist(self.df[self.data_column], # type: ignore
                bins, density=True, label='ratios histogram') 
        x = np.linspace(self.frozen_dist.ppf(0.01), 
                self.frozen_dist.ppf(0.99), 100)
        ax.plot(x, skewnorm.pdf(x, self.a, self.loc, self.scale),
                'r-', lw=3, alpha=0.6, label='skewnorm pdf')
        ax.legend(loc='best', frameon=False)
        plt.grid(True)
        plt.title('PDF Fitting')
        plt.show()


    def query(self, ratio):
        return self.frozen_dist.cdf(ratio + self.epsilon) \
                - self.frozen_dist.cdf(ratio - self.epsilon)


if __name__ == '__main__':
    fitter = PDFFitter('../static_data/data/C1_wpix_ratio.csv', 
            'C1', 'wpix_ratio', 0.05)
    fitter.plot_pdf_and_hist(0.5, 1.0, 0.03)
    fitter.print_info()

    fitter = PDFFitter('static_data/data/C2_wpix_ratio.csv', 
            'C2', 'wpix_ratio', 0.05)
    fitter.plot_pdf_and_hist(0.05, 1.0, 0.03)
    fitter.print_info()

    fitter = PDFFitter('static_data/data/C3_wpix_ratio.csv', 
            'C3', 'wpix_ratio', 0.05)
    fitter.plot_pdf_and_hist(0.05, 1.0, 0.03)
    fitter.print_info()
    
    fitter = PDFFitter('static_data/data/C4_wpix_ratio.csv', 
            'C4', 'wpix_ratio', 0.05)
    fitter.plot_pdf_and_hist(0.05, 1.0, 0.03)
    fitter.print_info()
    
    fitter = PDFFitter('static_data/data/C5_wpix_ratio.csv', 
            'C5', 'wpix_ratio', 0.05)
    fitter.plot_pdf_and_hist(0.05, 1.0, 0.03)
    fitter.print_info()
    """
    fitter = PDFFitter('../static_data/data/C1_shape.csv', 
            'C1', 'ratio', 0.02)
    fitter.plot_pdf_and_hist()
    fitter.print_info()
    fitter = PDFFitter('../static_data/data/C2_shape.csv',
            'C2','ratio', 0.02) 
    fitter.plot_pdf_and_hist()
    fitter.print_info()
    fitter = PDFFitter('../static_data/data/C3_shape.csv', 
            'C3','ratio', 0.02) 
    fitter.plot_pdf_and_hist()
    fitter.print_info()
    fitter = PDFFitter('../static_data/data/C4_shape.csv',
            'C4','ratio', 0.02) 
    fitter.plot_pdf_and_hist()
    fitter.print_info()
    fitter = PDFFitter('../static_data/data/C5_shape.csv',
            'C5','ratio', 0.02) 
    fitter.plot_pdf_and_hist()
    fitter.print_info()
    """
