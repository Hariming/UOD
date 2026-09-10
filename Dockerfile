FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace/Hari/uod_reference

COPY requirements.txt /tmp/uod-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/uod-requirements.txt

CMD ["bash"]
