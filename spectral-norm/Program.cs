using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace SpectralNorm
{
    class Program
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
            var u = new double[n + 3];
            var v = new double[n + 3];
            var tmp = new double[n + 3];

            u.AsSpan(0..n).Fill(1);

            for (var i = 0; i < 10; i++)
            {
                Mult_at_av(u, v, tmp, n);
                Mult_at_av(v, u, tmp, n);
            }

            return Math.Sqrt(Dot(u, v, n) / Dot(v, v, n));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double A(int i, int j)
        {
            return ((i + j) * (i + j + 1) >> 1) + i + 1;
        }

        private static double Dot(double[] v, double[] u, int n)
        {
            double sum = 0;
            for (var i = 0; i < n; i++)
                sum += v[i] * u[i];
            return sum;
        }

        private static void Mult_av(double[] v, double[] outv, int n)
        {
            Parallel.For(0, n, i =>
            {
                ref double v_ref = ref MemoryMarshal.GetArrayDataReference(v);
                Vector128<double> sum = Vector128<double>.Zero;
                for (var j = 0; j < n; j += 2)
                {
                    Vector128<double> b = LoadVector128(ref v_ref, j);
                    Vector128<double> a = Vector128.Create(A(i, j), A(i, j + 1));
                    sum = Sse2.Add(sum, Sse2.Divide(b, a));
                }

                Vector128<double> add = Sse3.HorizontalAdd(sum, sum);
                double value = Unsafe.As<Vector128<double>, double>(ref add);
                Unsafe.WriteUnaligned(ref Unsafe.As<double, byte>(ref GetArrayReference(outv, i)), value);
            });
        }

        private static void Mult_atv(double[] v, double[] outv, int n)
        {
            Parallel.For(0, n, i =>
            {
                ref double v_ref = ref MemoryMarshal.GetArrayDataReference(v);
                Vector128<double> sum = Vector128<double>.Zero;
                for (var j = 0; j < n; j += 2)
                {
                    Vector128<double> b = LoadVector128(ref v_ref, j);
                    Vector128<double> a = Vector128.Create(A(j, i), A(j + 1, i));
                    sum = Sse2.Add(sum, Sse2.Divide(b, a));
                }

                Vector128<double> add = Sse3.HorizontalAdd(sum, sum);
                double value = Unsafe.As<Vector128<double>, double>(ref add);
                Unsafe.WriteUnaligned(ref Unsafe.As<double, byte>(ref GetArrayReference(outv, i)), value);
            });
        }

        private static void Mult_at_av(double[] v, double[] outv, double[] tmp, int n)
        {
            Mult_av(v, tmp, n);
            Mult_atv(tmp, outv, n);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ref T GetArrayReference<T>(T[] array, nint offset)
            => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(array), offset);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<double> LoadVector128(ref double start, nint offset)
            => Unsafe.ReadUnaligned<Vector128<double>>(ref Unsafe.As<double, byte>(ref Unsafe.Add(ref start, offset)));
    }
}