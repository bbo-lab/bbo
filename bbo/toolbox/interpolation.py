import numpy as np
from bbo import geometry
from scipy.interpolate import interp1d

def calc_time_offsets(traces_y, traces_t, align_range = (-np.inf, np.inf), test_space = None, iterations = 5,
                      distancefunction="euclidean-normalized"):

    if test_space is None:
        test_space = (traces_t[0], traces_t[-1], 1000)

    # Aligns 0 values of traces_t such that distance of traces_y is minimized according to distancefunction. Final 0
    # value is initialized to traces_t[0], but eventually arbitrary.
    assert all([len(y) == len(t) for y, t in zip(traces_y, traces_t)]), "Trace t and y must match"
    timeoffsets = np.zeros(shape=(len(traces_y)))
    testoffsets = np.linspace(*test_space)

    allazimuth = traces_y[0]
    alltimes = traces_t[0]
    for i in range(iterations):
        for i_trace, (y, t) in enumerate(zip(traces_y, traces_t)):
            t = t - timeoffsets[i_trace]

            mask = (align_range[0] <= t) & (t <= align_range[1])

            correlation = cross_similarity(y[mask], t[mask],
                                           allazimuth, alltimes, testoffsets, min_exist=5,
                                           distancefunction=distancefunction)

            timeoffsets[i_trace] += testoffsets[np.argmin(correlation)]

        allazimuth, variance, alltimes = get_average_signal(traces_y, traces_t, timeoffsets=timeoffsets, timewindow=np.max(np.abs(align_range)))
    return timeoffsets


def align_by_logistic(traces_y, traces_t, align_range = (-np.inf, np.inf), return_fits=False):
    from scipy.optimize import curve_fit

    def logistic(xdata, L, x0, k, b):
        ydata = L / (1 + np.exp(-k * (xdata - x0))) + b
        return (ydata)

    timeoffsets = []
    logistics = []
    for y, t in zip(traces_y, traces_t):
        mask = (align_range[0] <= t) & (t <= align_range[1]) & ~np.isnan(y)

        k0 = (y[-1]-y[0])/(t[-1]-t[0]) / (max(y)-min(y))
        p0 = [max(y)-min(y), np.median(t), k0, min(y)]
        try:
            popt, pcov = curve_fit(logistic, t[mask], y[mask], p0, method='dogbox')
        except RuntimeError:
            popt = np.array([np.nan, np.nan, np.nan, np.nan])

        timeoffsets.append(popt[1])
        if return_fits:
            logistics.append(logistic(t, *popt))

    if return_fits:
        return np.array(timeoffsets), logistics
    else:
        return np.array(timeoffsets)


def get_average_signal(traces_y, traces_t, all_times=None, timeoffsets=None, value_offsets=None, timewindow=0.1, sigma=0.0, method="interpolate", only_valid=True):
    if value_offsets is None:
        value_offsets = np.zeros(shape=len(traces_y))
    if timeoffsets is None:
        timeoffsets = np.zeros(shape=len(traces_y))
    match method:
        case "direct":
            all_times = []
            all_data = []
            for d, t, t0, v0 in zip(traces_y, traces_t, timeoffsets, value_offsets):
                begin = np.searchsorted(t, t0 - timewindow)
                end = np.searchsorted(t, t0 + timewindow)
                all_times.append(t[begin:end] - t0)
                all_data.append(d[begin:end] - v0)
            all_times = np.concatenate(all_times)
            all_data = np.concatenate(all_data)
            sortindices = np.argsort(all_times)
            sortindices = sortindices[~np.isnan(all_data[sortindices])]
            all_times = all_times[sortindices]
            all_data = all_data[sortindices]
            all_data = geometry.smooth(all_data, all_times, sigma=sigma, num_neighbors=50)
            select_times = np.unique(all_times, return_index=True)[1]
            return all_data[select_times], all_times[select_times], None
        case "interpolate":
            all_times = []
            for d, t, t0, v0 in zip(traces_y, traces_t, timeoffsets, value_offsets):
                begin = np.searchsorted(t, t0 - timewindow)
                end = np.searchsorted(t, t0 + timewindow)
                all_times.append(t[begin:end] - t0)
            all_times = np.unique(np.concatenate(all_times))
            average = []
            for d, t, t0, v0 in zip(traces_y, traces_t, timeoffsets, value_offsets):
                if only_valid:
                    valid = np.isfinite(d)
                    t = t[valid]
                    d = d[valid]
                average.append(interp1d(t - t0, d - v0, bounds_error=False, fill_value=np.nan)(all_times))
            average = np.asarray(average)
            variance = np.nanvar(average, axis=0)
            average = np.nanmean(average, axis=0)
            average = geometry.smooth(average, all_times, sigma=sigma, num_neighbors=50)
            return average, variance, all_times
        case _: raise NotImplementedError(f"Method {method} not implemented")


def cross_similarity(data0, times0, data1, times1, timediffs, min_exist=None, distancefunction="euclidean"):
    similarities = np.full(shape=timediffs.shape, dtype=float, fill_value=np.inf)
    if data0.ndim == 2:
        data1 = list(np.moveaxis(data1, -1,0))

    for i in range(len(timediffs)):
        if data1.ndim == 1:
            data1_interp = np.interp(times0 - timediffs[i], times1, data1)
        else:
            data1_interp = np.stack([np.interp(times0 - timediffs[i], times1, data1[dim]) for dim in
                                   range(times1.shape[1])], axis=1)
        difference = data0 - data1_interp
        difference = difference[~np.isnan(difference)]

        if distancefunction == "euclidean":
            similarity = np.sum(np.square(difference)) / len(difference)
        elif distancefunction == "euclidean-normalized":
            similarity = np.sum(np.square(difference - np.mean(difference))) / len(difference)
        else:
            raise Exception(f"Unknown distance function {distancefunction}")

        if min_exist is None or len(difference) > min_exist:
            similarities[i] = similarity

    return similarities