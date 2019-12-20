/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>
// ibmGBT
#include <LightGBM/utils/log.h>
#include <LightGBM/config.h>

#include <iostream>

// ibmGBT
#include <fstream>
#include <chrono>
#include <string>

int main(int argc, char** argv) {
  try {
    // ibmGBT
    std::chrono::duration<double, std::milli> main_time;
    auto start_main_time = std::chrono::steady_clock::now();

    LightGBM::Application app(argc, argv);
    app.Run();

    // ibmGBT
    main_time = std::chrono::steady_clock::now() - start_main_time;
    LightGBM::Log::Info("main::main time: %f", main_time * 1e-3);
    #ifdef TIMETAG
    LightGBM::Config config = app.GetConfig();
    if ( config.log_train_time ){
      std::string log_filepath = config.log_folder + "ibmGBT_main_time.txt";
      std::ofstream resultfile(log_filepath.c_str(), std::ofstream::app);
      resultfile << config.device_type << "     ";
      resultfile << config.objective << "       ";
      resultfile << main_time.count() * 1e-3 << "\n";
      resultfile.close();
    }
    #endif
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
}
