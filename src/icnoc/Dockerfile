FROM auto_test:v1.0

RUN rm -rf /auto_test
ADD auto_test /auto_test
WORKDIR /auto_test

ENTRYPOINT ["/bin/bash", "/auto_test/start.sh"]
