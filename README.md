<div align="center">
  <img src="/_media/car_shapes.png" width="800" height="500">
</div>
 
# ML App for Car Price Calculation
A toy AI project that predicts a car price using an AI model trained on a [public automobile dataset](https://www.kaggle.com/datasets/toramky/automobile-dataset/). The idea is people who want to buy a car of a certain brand to select some of its parameters and to see how much money will cost them. The App is deployed in the cloud and everyone can use it as a preliminary market research tool. 

Besides AI algorithms we apply statistic component analysis to determine precisely the price of the car the user wants to see.

<div align="center">
  <img src="/_media/gui_test_car_attributes.png" width="700" height="800">
</div>

### Installation, Setup and 

To run the project in your local system, you need to install the project first:

`git clone https://github.com/vladimirkanchev/auto-price`

`cd auto-price`

`pip install -r requirements.txt`

Then you can run the App locally:

`streamlit run src/main`


### Technologies

    At the current moment we use the following software technologies:
    * Visual Studio Code 1.82.2
    * Python 3.10.12
    * Streamlit 1.27.1
    * Streamlit Community cloud
    
### Python Packages Used

    Some of the python packages which are part of our project:
    * Numpy 1.23.5
    * Pandas 2.0.1
    * Prince 0.12.1 (Component Analysis)
    * Seaborn 0.12.2 
    * Sklearn 1.2.2
   
    
### Data
    
    While at the current moment we use only the above mentioned public car dataset, we plan to use larger car datasets in the future. We appreciate all suggestions for other publicly availably car datasets. 
    
### Code Structure
   
   Currently our project has the following structure:
   
   <img src="/_media/project_tree.png" width="400" height="500">  


### Results and Evaluations
    At the current moment we have implemented only Linear Regression prediction algorithm. We are aware we need a solid benchmark car dataset for  evaluation.
    
### Future Work
    * Add better regression algorithms as Gradient Boosting, etc.
    * Improve component analysis algorithms
    * Speed up the ML algorithms and decrease memory requirements.
    * Implement a simple CI/CD workflow and MLOPS
    
### Who wants to contribute
    Contributions, issues and feature requests will be welcomed at the later stage of project development. 

### References

https://github.com/vladimirkanchev/auto-pricehttps://github.com/vladimirkanchev/auto-price
