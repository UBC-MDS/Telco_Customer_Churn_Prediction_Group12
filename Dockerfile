# Docker file for the telco customer churn predictor project
# Adam Morphy, Dec, 2021

FROM rocker/tidyverse 

# install python
RUN apt-get update && apt-get install -y python3 python3-pip

# install libxt6
RUN apt-get install -y --no-install-recommends libxt6

# install the kableExtra package using install.packages
RUN Rscript -e "install.packages('kableExtra')"

# install knitr
RUN Rscript -e "install.packages('knitr', dependencies = TRUE)"


# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put miniconda in path so we can use conda
ENV PATH=$CONDA_DIR/bin:$PATH

# install python packages
RUN conda install docopt \
    docopt \
    requests \
    ipykernel \
    matplotlib>=3.2.2 \
    scikit-learn>=1.0 \
    pandas>=1.3.* \
    altair \
    jsonschema=3.2.0 \
    seaborn \
    pip \
    numpy


# Install packages need for saving plots properly
RUN conda install -c conda-forge vega-cli vega-lite-cli

RUN pip install dataframe_image

RUN pip install altair-saver

RUN pip install lxml
