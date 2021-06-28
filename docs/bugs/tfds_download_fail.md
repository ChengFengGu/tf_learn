> All attempts to get a Google authentication bearer token failed, returning an empty token. Retrieving token from files failed with "Not found: Could not locate the credentials file.". Retrieving token from GCE failed with "Failed precondition: Error executing an HTTP request: libcurl code 6 meaning 'Couldn't resolve host name', error details: Couldn't resolve host 'metadata'".


resolve:


```shell

TFDS_HTTP_PROXY=http://127.0.0.1:8889 TFDS_HTTPS_PROXY=http://127.0.0.1:8889 TFDS_FTP_PROXY=http://127.0.0.1:8889 python classify/tf_hub_nlp_classify.py 

```