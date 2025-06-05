FROM artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.torch-gpu 

RUN pip3 install toml scikit-image

RUN git clone https://github.com/quic/aimet.git
WORKDIR /aimet/
RUN mkdir build

WORKDIR /aimet/build/
RUN cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_TENSORFLOW=OFF
RUN make -j8 
RUN make install
ENV PYTHONPATH=/aimet/build/staging/universal/lib/python:$PYTHONPATH

ENTRYPOINT [ "python3" ]
