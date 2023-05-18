FROM zhaosheng/vaf_clean:v0.5.0
ADD src.tar.gz /VAF
WORKDIR /VAF/src
RUN chmod +x /VAF/src/start.sh
ENTRYPOINT ["/VAF/src/start.sh"]
