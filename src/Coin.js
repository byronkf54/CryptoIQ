import React, { useState } from 'react';
import './CSS/Coin.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCircleInfo } from '@fortawesome/free-solid-svg-icons';

const Coin = ({name, image, symbol, marketcap, volume, price, pricechange, rating}) => {
    const [showTooltip, setShowTooltip] = useState(false);

    return (
        <tr>
            <td>
              <div className="coin">
                <div className="coin-details">
                  <img src={image} alt="crypto" />
                  <div>
                    <h1>{name}</h1>
                    <p className="coin-symbol">{symbol}</p>
                  </div>
                </div>
              </div>
            </td>
            <td>£{price}</td>
            <td>£{volume.toLocaleString()}</td>
            <td>
                {pricechange < 0 ? (
                    <p className="coin-percent red">{pricechange.toFixed(2)}%</p>
                ) : (
                    <p className="coin-percent green">{pricechange.toFixed(2)}%</p>
                )}
            </td>
            <td>£{marketcap.toLocaleString()}</td>
            <td style={{ position: 'relative' }}>
                {rating !== undefined ? (
                    <>
                        {rating}
                        <FontAwesomeIcon
                            icon={faCircleInfo}
                            style={{ position: 'absolute', right: 0, cursor: 'pointer' }}
                            onClick={() => setShowTooltip(!showTooltip)}
                        />
                        {showTooltip && (
                        <div
                            style={{
                            position: 'absolute',
                            top: '0px',
                            left: 'calc(100% + 5px)',
                            backgroundColor: 'white',
                            border: '1px solid black',
                            padding: '4px',
                            }}
                        >
                        Explanation...
                      </div>
                    )}
                  </>
                ) : 'TBD'}

            </td>
    </tr>
    )
}

export default Coin
