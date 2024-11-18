# Use a Miniconda base image
FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-lang-all \
    dvipng \
    cm-super \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# # RUN apt-get update && apt-get install -y --no-install-recommends install cm-super
# # Update the package list and install cm-super
# RUN apt-get update && apt-get install -y --no-install-recommends cm-super \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy environment.yml (optional, if you want to predefine an environment)
COPY environment.yml .

# Create and activate the Conda environment
RUN conda env create -f environment.yml \
    && echo "conda activate quad-stable-opinf" >> ~/.bashrc

# Activate the environment
SHELL ["conda", "run", "-n", "quad-stable-opinf", "/bin/bash", "-c"]

# Set the default shell to bash with Conda activated
SHELL ["/bin/bash", "--login", "-c"]

# 7. Create a volume for Poetry cache to persist between builds
VOLUME ["/root/.cache/pypoetry"]

# 9. Copy only the dependency files first for better layer caching
COPY pyproject.toml poetry.lock ./

# 8. Leverage the Poetry cache during dependency installation
RUN poetry config cache-dir /root/.cache/pypoetry \
    && poetry install --no-root

COPY ./src ./src
COPY ./Examples ./Examples
COPY ./README.md ./ 
COPY ./run_examples.sh ./

RUN poetry install

# Set the default command
CMD ["bash"]