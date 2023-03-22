FROM q935970314/mialgo:2022

COPY . ./workdir/
COPY ./run.sh /run.sh

WORKDIR /workdir/stage5
RUN export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && pip install -r requirements.txt && python setup.py develop

WORKDIR /


