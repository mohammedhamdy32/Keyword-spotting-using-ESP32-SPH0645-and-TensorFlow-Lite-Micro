/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Basic class for computing MFCCs from spectrogram slices.

#ifndef TENSORFLOW_CORE_KERNELS_MFCC_H_
#define TENSORFLOW_CORE_KERNELS_MFCC_H_

#include <vector>

//#include "tensorflow/core/framework/op_kernel.h"
#include "mfcc_dct.h"
#include "mfcc_mel_filterbank.h"
// #include "tensorflow/core/platform/logging.h"
#include <iostream>

namespace tensorflow {

class Mfcc {
 public:
  Mfcc();
  bool Initialize(int input_length, 
                  double input_sample_rate,
                  double lower_freq_limit,
                  double upper_freq_limit,
                  int filterbank_channel_count,
                  int dct_coefficient_count,
                  double kFilterbankFloor);

  // Input is a single squared-magnitude spectrogram frame. The input spectrum
  // is converted to linear magnitude and weighted into bands using a
  // triangular mel filterbank, and a discrete cosine transform (DCT) of the
  // values is taken. Output is populated with the lowest dct_coefficient_count
  // of these values.
  void Compute(const std::vector<double>& spectrogram_frame,
               std::vector<double>* output) const;

  void set_upper_frequency_limit(double upper_frequency_limit) {
    if (!initialized_){
      std::cout << "Set frequency limits before calling Initialize.";
      exit(-1);
    }

    upper_frequency_limit_ = upper_frequency_limit;
  }

  void set_lower_frequency_limit(double lower_frequency_limit) {
    if (!initialized_){
      std::cout  << "Set frequency limits before calling Initialize.";
      exit(-1);
    }
    lower_frequency_limit_ = lower_frequency_limit;
  }

  void set_filterbank_channel_count(int filterbank_channel_count) {
    if (!initialized_){
      std::cout << "Set channel count before calling Initialize.";
      exit(-1);
    }
    filterbank_channel_count_ = filterbank_channel_count;
  }

  void set_dct_coefficient_count(int dct_coefficient_count) {
    if (!initialized_){
      std::cout << "Set coefficient count before calling Initialize.";
      exit(-1);
    }
    dct_coefficient_count_ = dct_coefficient_count;
  }

 private:
  MfccMelFilterbank mel_filterbank_;
  MfccDct dct_;
  bool initialized_;
  double lower_frequency_limit_;
  double upper_frequency_limit_;
  int filterbank_channel_count_;
  int dct_coefficient_count_;
  double kFilterbankFloor_;
  TF_DISALLOW_COPY_AND_ASSIGN(Mfcc);
};

}  // namespace tensorflow

extern tensorflow::Mfcc mfcc;

#endif  // TENSORFLOW_CORE_KERNELS_MFCC_H_