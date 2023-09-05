extern crate rustaccel;
extern crate rustaccel_cuda;

use rustaccel::prelude::*;
use rustaccel_cuda::prelude::*;
use std::time::Instant;

fn main() {
    // 获取所有可用的CUDA设备
    let devices = Device::list_cuda_devices();

    // 创建一个Rayon线程池，以便在多个GPU上并行执行任务
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(devices.len())
        .build()
        .unwrap();

    // 矩阵大小
    let size = 1024;

    // 创建两个随机矩阵
    let a = Matrix::random(&queue, size, size);
    let b = Matrix::random(&queue, size, size);

    // 记录开始时间
    let start_time = Instant::now();

    // 并行执行矩阵乘法
    let results: Vec<_> = pool.install(|| {
        devices
            .par_iter()
            .map(|device| {
                // 创建一个计算上下文和一个计算队列
                let context = Context::new(&device);
                let queue = Queue::new(&context, &device);

                // 创建一个用于存储结果的矩阵
                let mut c = Matrix::zero(&queue, size, size);

                // 执行矩阵乘法
                gemm(&queue, &a, &b, &mut c, 1.0, 0.0);

                // 返回执行时间
                let elapsed_time = start_time.elapsed();
                elapsed_time.as_millis()
            })
            .collect()
    });

    // 输出每个GPU的执行时间
    for (i, result) in results.iter().enumerate() {
        println!("GPU {} execution time: {} ms", i, result);
    }
}

fn gemm(queue: &Queue, a: &Matrix<f64>, b: &Matrix<f64>, c: &mut Matrix<f64>, alpha: f64, beta: f64) {
    // 获取矩阵的大小
    let m = a.rows();
    let n = b.cols();
    let k = a.cols();

    // 创建CUDA内核，用于执行矩阵乘法
    let kernel_code = format!(
        r#"
        __global__ void gemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {{
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < M && col < N) {{
                float sum = 0.0;
                for (int i = 0; i < K; ++i) {{
                    sum += A[row * K + i] * B[i * N + col];
                }}
                C[row * N + col] = alpha * sum + beta * C[row * N + col];
            }}
        }}
        "#
    );

    let kernel = Kernel::create(&queue, &kernel_code, "gemm").unwrap();

    // 设置CUDA内核的参数
    kernel.set_arg(0, a.as_ptr());
    kernel.set_arg(1, b.as_ptr());
    kernel.set_arg(2, c.as_mut_ptr());
    kernel.set_arg(3, m);
    kernel.set_arg(4, n);
    kernel.set_arg(5, k);
    kernel.set_arg(6, alpha);
    kernel.set_arg(7, beta);

    // 定义CUDA内核的工作尺寸
    let work_size = WorkSize::new((n as u32, m as u32));

    // 执行CUDA内核
    queue.enqueue_kernel(&kernel, work_size);
    queue.finish().unwrap();
}
