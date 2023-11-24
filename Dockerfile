FROM python:3.10 AS build

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


COPY ./app /app

WORKDIR /app

FROM build AS test
RUN pip3 install pytest && rm -rf /root/.cache
RUN pytest

FROM build AS final
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1313"]

