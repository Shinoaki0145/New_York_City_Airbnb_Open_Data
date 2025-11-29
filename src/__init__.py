"""
Package chứa các module cho dự án phân tích dữ liệu Airbnb NYC 2019

Modules:
--------
- data_processing: Các hàm xử lý và làm sạch dữ liệu
- visualization: Các hàm trực quan hóa dữ liệu sử dụng matplotlib và seaborn
- models: Các mô hình machine learning
"""

from .visualization import *
from .data_processing import *
from .models import *

__version__ = '1.0.0'
__author__ = 'Shinoaki'
