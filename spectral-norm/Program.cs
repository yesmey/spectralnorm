using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

namespace SpectralNorm
{
    unsafe class Program
    {
        public static void Main(string[] args)
        {
            int n = 8000;
            if (args.Length > 0) n = int.Parse(args[0]);

            var answer = Spectralnorm(n);
            Console.WriteLine("{0:f9}", answer);
        }

        private static double Spectralnorm(int n)
        {
            double* u = (double*)NativeMemory.AlignedAlloc((nuint)(n + Vector128<double>.Count) * sizeof(double), (nuint)Vector128<byte>.Count);
            double* v = (double*)NativeMemory.AlignedAlloc((nuint)(n + Vector128<double>.Count) * sizeof(double), (nuint)Vector128<byte>.Count);
            double* tmp = (double*)NativeMemory.AlignedAlloc((nuint)(n + Vector128<double>.Count) * sizeof(double), (nuint)Vector128<byte>.Count);

            new Span<double>(u, n).Fill(1);

            for (var i = 0; i < 10; i++)
            {
                Mult_at_av(u, v, tmp, n);
                Mult_at_av(v, u, tmp, n);
            }

            double result = Math.Sqrt(Dot(u, v, n) / Dot(v, v, n));

            NativeMemory.AlignedFree(u);
            NativeMemory.AlignedFree(v);
            NativeMemory.AlignedFree(tmp);

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double A(int i, int j)
        {
            return ((i + j) * (i + j + 1) >> 1) + i + 1;
        }

        private static double Dot(double* v, double* u, int n)
        {
            double sum = 0;
            for (var i = 0; i < n; i++)
                sum += v[i] * u[i];
            return sum;
        }

        private static void Mult_av(double* v, double* outv, int n)
        {
            Parallel.For(0, n, i =>
            {
                Vector128<double> sum = Vector128<double>.Zero;
                for (int j = 0; j < n; j += Vector128<double>.Count)
                {
                    Vector128<double> b = Vector128.LoadAligned(v + j);
                    Vector128<double> a = Vector128.Create(A(i, j), A(i, j + 1));
                    sum = Vector128.Add(sum, Vector128.Divide(b, a));
                }

                double value = Vector128.Sum(sum);
                Unsafe.WriteUnaligned(outv + i, value);
            });
        }

        private static void Mult_atv(double* v, double* outv, int n)
        {
            Parallel.For(0, n, i =>
            {
                Vector128<double> sum = Vector128<double>.Zero;
                for (int j = 0; j < n; j += Vector128<double>.Count)
                {
                    Vector128<double> b = Vector128.LoadAligned(v + j);
                    Vector128<double> a = Vector128.Create(A(j, i), A(j + 1, i));
                    sum = Vector128.Add(sum, Vector128.Divide(b, a));
                }

                double value = Vector128.Sum(sum);
                Unsafe.WriteUnaligned(outv + i, value);
            });
        }

        private static void Mult_at_av(double* v, double* outv, double* tmp, int n)
        {
            Mult_av(v, tmp, n);
            Mult_atv(tmp, outv, n);
        }
    }
}