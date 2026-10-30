using NumSharp;
using System.Runtime.CompilerServices;

namespace FastDtwSharp
{
    public class Utility
    {
        public static long[] GetRange(long startNumber, long endNumber, int step = 1)
        {
            List<long> numbers = new List<long>();

            for (var i = startNumber; i < endNumber;)
            {
                numbers.Add(i);

                i += step;
            }

            return numbers.ToArray();
        }

        public static int GetLength(dynamic obj)
        {
            if (obj is null)
            {
                return 0;
            }

            if (obj is NDArray nd)
            {
                return GetNDArrayLength(nd);
            }
            else if (IsList(obj))
            {
                return obj.Count;
            }
            else if (IsTuple(obj))
            {
                return (obj as ITuple).Length;
            }

            return 0;
        }

        public static bool IsTuple(object obj)
        {
            if (obj is null)
            {
                return false;
            }

            Type type = obj.GetType();

            return type.IsGenericType && type.GetGenericTypeDefinition().Name.StartsWith("ValueTuple`");
        }

        public static bool IsList(object obj)
        {
            if (obj is null)
            {
                return false;
            }

            Type type = obj.GetType();

            return type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>);
        }

        public static int GetNDArrayLength(NDArray array)
        {
            return array is null ? 0 : (int)array.__len__();
        }

        public static float[,] GetArrayFromNdArrayList(List<NDArray> list)
        {
            float[,] result = new float[list.Count, GetLength(list[0])];

            for(var i=0;i< result.GetLength(0);i++)
            {
                for(var j=0;j<result.GetLength(1);j++)
                {
                    result[i, j] = list[i][j].item<float>();
                }
            }

            return result;
        }

        public static float[][] GetArrayFromNdArrayList2(List<NDArray> list)
        {
            float[][] result = new float[list.Count][];

            for (var i = 0; i < result.GetLength(0); i++)
            {
                result[i] = new float[result.GetLength(1)];

                for (var j = 0; j < result.GetLength(1); j++)
                {
                    result[i][j] = list[i][j].item<float>();
                }
            }

            return result;
        }
    }
}
