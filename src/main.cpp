#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>
#include <libutils/string_utils.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace utils;

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

std::vector<cl_platform_id> get_platforms() {
    cl_uint num_platforms;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &num_platforms));

    std::vector<cl_platform_id> platforms(num_platforms);
    OCL_SAFE_CALL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    return platforms;
}

std::vector<cl_device_id> get_devices(const cl_platform_id platform, cl_device_type type) {
    cl_uint num_devices;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, type, 0, nullptr, &num_devices));

    std::vector<cl_device_id> devices(num_devices);
    if (num_devices)
        OCL_SAFE_CALL(clGetDeviceIDs(platform, type, num_devices, devices.data(), nullptr));

    return devices;
}

std::string get_platform_name(const cl_platform_id platform) {
    size_t name_size = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &name_size));

    std::vector<unsigned char> platformName(name_size, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, name_size, platformName.data(), nullptr));

    return std::string(platformName.begin(), platformName.end());
}

std::string get_device_name(const cl_device_id device) {
    size_t name_size = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &name_size));

    std::vector<unsigned char> device_name(name_size, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, name_size, device_name.data(), nullptr));

    return std::string(device_name.begin(), device_name.end());
}

cl_device_id select_gpu(const std::vector<cl_platform_id>& platforms, std::string platform_name) {
    for (const cl_platform_id platform : platforms) {
        auto devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
        if (devices.empty())
            continue;
        
        for (const cl_device_id device : devices) {
            std::string name = get_device_name(device);
            name = utils::tolower(name);

            if (name.find(platform_name) != std::string::npos) {
                return device;
            }
        }
    }

    return nullptr;
    //todo: add code to return cpu device!
}

std::vector<char> get_program_build_log(const cl_program program, const cl_device_id device) {
    const size_t params_size = 1;
    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));

    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    return log;
}

void CL_CALLBACK cl_notify_build_ctx_error(const char* errinfo, const void* private_info, size_t cb, void* user_data) {
    std::stringstream ss;
    ss << "err-info: " << errinfo << "\n" << "private-info: " << private_info << std::endl;
   
    throw std::runtime_error(ss.str());
}

cl_context create_context(const cl_device_id device) {
    cl_int err_code = 0;
    cl_context context = clCreateContext(nullptr, 1, &device, &cl_notify_build_ctx_error, nullptr, &err_code);
    return context;
} 

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    std::vector<cl_platform_id> platforms = get_platforms();
    cl_device_id device = select_gpu(platforms, "nvidia");

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_context context = create_context(device);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    cl_int err_code = 0;
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err_code);
    OCL_SAFE_CALL(err_code);

    unsigned int n = 1000 * 1000 * 450;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    cl_mem as_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, as.size() * sizeof(float), 
                                    as.data(), &err_code);
    OCL_SAFE_CALL(err_code);

    cl_mem bs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bs.size() * sizeof(float),
                                      bs.data(), &err_code);
    OCL_SAFE_CALL(err_code);

    cl_mem cs_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, cs.size() * sizeof(float),
                                    nullptr, &err_code);
    OCL_SAFE_CALL(err_code);


    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        //std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    const char *code = kernel_sources.c_str();
    const size_t lengths[] = { kernel_sources.size() };
    cl_program program = clCreateProgramWithSource(context, 1, &code, lengths, &err_code);
    OCL_SAFE_CALL(err_code);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    std::vector<char> log = get_program_build_log(program, device);
    if (log.size() > 2) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    } else {
        std::cout << "Compiled successfuly." << std::endl;
    }
    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    cl_kernel aplusb_kernel = clCreateKernel(program, "aplusb\0", &err_code);
    OCL_SAFE_CALL(err_code);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        size_t i = 0;
        OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, i++, sizeof(cl_mem), (void *)(& as_buffer)));
        OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, i++, sizeof(cl_mem), (void *)(&bs_buffer)));
        OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, i++, sizeof(cl_mem), (void *)(&cs_buffer)));
        OCL_SAFE_CALL(clSetKernelArg(aplusb_kernel, i++, sizeof(unsigned int), (void *)(&n)));
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        std::cout << "Start working" << std::endl;
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, aplusb_kernel, 1, nullptr, &global_work_size,
                                                 &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));

            // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
            t.nextLap();
        }
        std::cout << "End working" << std::endl;

        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        uint64_t a = 3 * n * sizeof(float);
        std::cout << "VRAM bandwidth: " << a / t.lapAvg() / (1024*1024*1024) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            clEnqueueReadBuffer(command_queue, cs_buffer, CL_TRUE, 0, cs.size() * sizeof(float), cs.data(), 0, nullptr,
                                nullptr);
            t.nextLap();
        }
        uint64_t b = n * sizeof(float);
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << b / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

//     TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(aplusb_kernel));
    OCL_SAFE_CALL(clReleaseMemObject(cs_buffer));
    OCL_SAFE_CALL(clReleaseMemObject(bs_buffer));
    OCL_SAFE_CALL(clReleaseMemObject(as_buffer));
    OCL_SAFE_CALL(clReleaseCommandQueue(command_queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
