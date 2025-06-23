From sgl


RUN git clone https://github.com/swiss-ai/transformers.git && \
    cd transformers && \
    git checkout main && \
    python -m pip install -e . && \
    cd ..


RUN git clone https://github.com/swiss-ai/lm-evaluation-harness.git && \
    cd lm-evaluation-harness && \
    # git checkout new-include && \
    git checkout merge-upstream && \
    python -m pip install -e .[api,longbench,ruler,wandb] && \
    cd ..
