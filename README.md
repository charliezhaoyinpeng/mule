# mule

## Requirements
Some key dependencies are listed below, while others are given in [`requirements.txt`](https://github.com/Siyu-C/ACAR-Net/blob/master/requirements.txt).
- Python >= 3.6
- PyTorch >= 1.3, and a corresponding version of torchvision
- ffmpeg (used in data preparation)
- Download pre-trained models, which are listed in [`pretrained/README.md`](https://github.com/Siyu-C/ACAR-Net/blob/master/pretrained/README.md), to the `pretrained` folder.
- Prepare data. Please refer to [`DATA.md`](https://github.com/Siyu-C/ACAR-Net/blob/master/DATA.md).
- Download annotations files to the `annotations` folder. See [`annotations/README.md`](https://github.com/Siyu-C/ACAR-Net/blob/master/annotations/README.md) for detailed information.

## Usage
Default values for arguments `nproc_per_node`, `backend` and `master_port` are `8`, `nccl` and `31114` respectively.

```
python main.py --config CONFIG_FILE [--nproc_per_node N_PROCESSES] [--backend BACKEND] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]
```

### Running with Multiple Machines
In this case, the `master_addr` argument must be provided. Moreover, arguments `nnodes` and `node_rank` can be additionally specified (similar to `torch.distributed.launch`), otherwise the program will try to obtain their values from environment variables. See [`distributed_utils.py`](https://github.com/Siyu-C/ACAR-Net/blob/master/distributed_utils.py) for details.


## About Our Paper
![architecture-fig]

[architecture-fig]: https://github.com/charliezhaoyinpeng/mule/tree/main/figs/architecture.png "architecture"

Please cite with the following Bibtex code:

```
@inproceedings{zhao2023open,
  title={Open Set Action Recognition via Multi-Label Evidential Learning},
  author={Zhao, Chen and Du, Dawei and Hoogs, Anthony and Funk, Christopher},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Zhao, Chen, Du, Dawei, Hoogs, Anthony and Funk, Christopher. "Open Set Action Recognition via Multi-Label Evidential Learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.*
