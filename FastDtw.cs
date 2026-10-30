using NumSharp;

namespace FastDtwSharp
{
    public class FastDtw
    {
        /// <summary>
        /// return the approximate distance between 2 time series with O(N)
        /// time and memory complexity
        /// </summary>
        /// <param name="x">array_like</param>
        /// <param name="y">array_like</param>
        /// <param name="radius">
        /// size of neighborhood when expanding the path. A higher value will
        /// increase the accuracy of the calculation but also increase time
        /// and memory consumption. A radius equal to the size of x and y will
        /// yield an exact dynamic time warping calculation.
        /// </param>
        /// <param name="dist">
        /// The method for calculating the distance between x[i] and y[j]. If
        /// dist is an int of value p > 0, then the p-norm will be used. If
        /// dist is a function then dist(x[i], y[j]) will be used. If dist is
        /// None then abs(x[i] - y[j]) will be used.
        /// </param>
        /// <returns>
        /// distance: the approximate distance between the 2 time series
        /// path:list of indexes for the inputs x and y
        /// </returns>
        public static (float distance, List<(long,long)> path) fastdtw(NDArray x, NDArray y, int radius = 1, Delegate dist = null)
        {
            var result = __prep_inputs(x, y, dist);

            x = result.Item1;
            y = result.Item2;
            dist = result.Item3;

            return __fastdtw(x, y, radius, dist);
        }

        private static (float distance, List<(long, long)> path) __fastdtw(dynamic x, dynamic y, int radius, Delegate dist)
        {
            var min_time_size = radius + 2;

            var x_len = Utility.GetLength(x);
            var y_len = Utility.GetLength(y);

            if (x_len < min_time_size || y_len < min_time_size)
            {
                return dtw(x, y, dist: dist);
            }

            var x_shrinked = __reduce_by_half(x);
            var y_shrinked = __reduce_by_half(y);

            var result = __fastdtw(x_shrinked, y_shrinked, radius: radius, dist: dist);
            var distance = result.Item1;
            var path = result.Item2;

            var window = __expand_window(path, x_len, y_len, radius);

            return __dtw(x, y, window, dist: dist);
        }

        private static (float, List<(long, long)>) __dtw(dynamic x, dynamic y, List<(float, float)> window, Delegate dist)
        {
            var len_x = Utility.GetLength(x);
            var len_y = Utility.GetLength(y);

            if (window is null)
            {
                window = new List<(float, float)>();

                foreach(var i in Utility.GetRange(0, len_x))
                {
                    foreach(var j in Utility.GetRange(0, len_y))
                    {
                        window.Add((i, j));
                    }
                }                
            }

            window = window.Select(item => (item.Item1 + 1, item.Item2 + 1)).ToList();

            var  D = new Dictionary<(long,long),(float,long,long)> { { (0,0),(0, 0, 0) } };

            foreach (var item in window)
            {
                var i = (int)item.Item1;
                var j = (int)item.Item2;

                var dt = dist is not null?  Convert.ToSingle(dist.DynamicInvoke([x[i - 1], y[j - 1], null])): 0;

                D[(i, j)] = (new List<(float, int, int)>(){ (get_dict_item( D, (i - 1, j)).Item1 + dt, i - 1, j) , (get_dict_item(D,(i, j - 1)).Item1 + dt, i, j - 1),
                      (get_dict_item(D,(i - 1, j - 1)).Item1 + dt, i - 1, j - 1) }).OrderBy(item=>item.Item1).FirstOrDefault();
            }

            var path = new List<(long, long)>();

            long m = len_x;
            long n = len_y;

            while (!(m == 0 && n == 0))
            {
                path.Add((m - 1, n - 1));

                var item = get_dict_item(D, (m, n));

                m = item.Item2;
                n = item.Item3;
            }

            path.Reverse();

            return (get_dict_item(D,(len_x, len_y)).Item1, path);
        }

        private static (float, long, long) get_dict_item(Dictionary<(long, long), (float, long, long)> dict, (long,long) key)
        {
            if(!dict.ContainsKey(key))
            {
                dict[key] = (float.PositiveInfinity, 0, 0);
            }

            return dict[key];
        }

        /// <summary>
        /// return the distance between 2 time series without approximation
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="dist"></param>
        /// <returns></returns>
        public static dynamic dtw(dynamic x, dynamic y, dynamic dist = null)
        {
            var result = __prep_inputs(x, y, dist);

            x = result.Item1;
            y = result.Item2;
            dist = result.Item3;

            return __dtw(x, y, null, dist);
        }

        private static (NDArray, NDArray, dynamic) __prep_inputs(dynamic input_x, dynamic input_y, dynamic dist)
        {
            NDArray x = null; 
            NDArray y = null;

            if(input_x is NDArray nd)
            {
                x = np.asanyarray(nd, dtype: DType.Single);
            }
            else
            {
                float[,] array = Utility.GetArrayFromNdArrayList(input_x);

                x = np.asanyarray(array, dtype: DType.Single);
            }

            if(input_y is NDArray nd2)
            {
                y = np.asanyarray(nd2, dtype: DType.Single);
            }
            else
            {
                float[,] array = Utility.GetArrayFromNdArrayList(input_y);

                y = np.asanyarray(array, dtype: DType.Single);
            }

            if (x.ndim == y.ndim && x.ndim > 1 && x.shape[1] != y.shape[1])
            {
                throw new ValueError("second dimension of x and y must be the same");
            }

            if (dist is int && dist <= 0)
            {
                throw new ValueError("dist cannot be a negative integer");
            }

            if (dist is null)
            {
                if (x.ndim == 1)
                {
                    Delegate @delegate = __difference;

                    dist = @delegate;
                }
                else
                {
                    dist = __norm(p: 1);
                }
            }
            else if (dist is int)
            {
                dist = __norm(p: dist);
            }

            return (x, y, dist);
        }

        private static float __difference(float a, float b)
        {
            return Math.Abs(a - b);
        }

        private static Func<NDArray, NDArray, NDArray> __norm(NDArray p)
        {
            Func<NDArray, NDArray, NDArray> func = (a, b) =>
            {
                return np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p);
            };

            return func;
        }

        private static List<NDArray> __reduce_by_half(dynamic x)
        {
            var len = Utility.GetLength(x);

            NDArray nd = x as NDArray;
            var items = x as List<NDArray>;

            List<NDArray> list = new List<NDArray>();

            foreach(var i in Utility.GetRange(0, len - len % 2, 2))
            {
                if(nd is not null)
                {
                    list.Add((nd[i] + nd[1 + i]) / 2.0);
                }
                else
                {
                    list.Add((items[(int)i] + items[(int)(1 + i)]) / 2.0);
                }
            }    

            return list;
        }      

        private static dynamic __expand_window(List<(long, long)> path, long len_x, long len_y, int radius)
        {
            var path_ = path.ToHashSet();

            foreach (var (i, j) in path)
            {
                var items = Utility.GetRange(-radius, radius + 1).SelectMany(a => Utility.GetRange(-radius, radius + 1), (a, b) => (i + a, j + b));

                foreach (var (a, b) in items)
                {
                    path_.Add((a, b));
                }
            }

            var window_ = new HashSet<(float, float)>();

            foreach(var (i, j) in path_)
            {
                foreach(var (a, b) in new List<(float, float)>() { (i * 2, j * 2), (i * 2, j * 2 + 1), (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1) })
                {
                    window_.Add((a, b));
                }
            }

            var window = new List<(float, float)>();
            long start_j = 0l;
            long? new_start_j = null;

            foreach (var i in Utility.GetRange(0, len_x))
            {
                new_start_j = null;

                foreach (var j in Utility.GetRange(start_j, len_y))
                {
                    if(window_.Contains((i, j)))
                    {
                        window.Add((i, j));

                        if(new_start_j is null)
                        {
                            new_start_j = j;
                        }
                    }
                    else if(new_start_j is not null)
                    {
                        break;
                    }
                }

                start_j =new_start_j??0;
            }

            return window;
        }     
    }
}
